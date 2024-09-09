"""script for mistral and mixtral"""
from pathlib import Path
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
import random
import torch
import datasets
import huggingface_hub
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from kbqa.utils.train_eval import get_best_checkpoint_path


SEED = 13
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = ArgumentParser()
parser.add_argument(
    "--model_name",
    help="model name for mistral/mixtral",
    default="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
parser.add_argument(
    "--trained_model_path",
    help="trained model path for mixtral or mistral",
    default="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "eval", "train_eval"],
    help="Choose mode for working, train or evaluate/analyze fited model",
)
parser.add_argument(
    "--output_dir",
    help="output path for results",
    default="./mixtral_mintaka_checkpoints_4bit",
)
parser.add_argument(
    "--answer_candidates_output_path",
    default="generated_candidates/finetune_mistral_train.pkl",
    type=str,
    help="file path for the generated answer candidates",
)
parser.add_argument("--evaluation_split", default="test")

# prompt to feed mistral/mixtral
# pylint: disable=line-too-long
PROMPT = "<s>[INST] Answer as briefly as possible without additional information.\n{message}? [/INST]"


def convert_causal(my_tokenizer, source, target, max_length=256):
    """prepare causal formattings"""
    source_tokens = my_tokenizer(
        source,
        add_special_tokens=False,
        max_length=max_length,
        padding=False,
        truncation=True,
    )["input_ids"]
    if my_tokenizer.bos_token_id:
        source_tokens.insert(0, my_tokenizer.bos_token_id)
    input_ids = source_tokens[:]
    target_tokens = my_tokenizer(
        target,
        add_special_tokens=False,
        max_length=max_length,
        padding=False,
        truncation=True,
    )["input_ids"]
    input_ids += target_tokens + [my_tokenizer.eos_token_id]
    input_ids = torch.LongTensor(input_ids)
    labels = input_ids.clone()
    attention_mask = input_ids.new_ones(input_ids.size())
    labels[: len(source_tokens)] = -100
    assert (
        input_ids.size(0)
        == labels.size(0)
        == attention_mask.size(0)
        <= 2 * max_length + 2
    )

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def create_prompts(my_dataset, ds_split):
    """for answers generation, format the prompts for the split"""
    ques_prompts = []

    for row in range(len(my_dataset[ds_split])):
        ques_prompts.append(
            PROMPT.format(message=my_dataset[ds_split][row]["question"])
        )
    return ques_prompts


def process_data(my_dataset, my_tokenizer):
    """process data for training by creating prompt for llm"""
    proc_dataset = {}
    for split in ["train", "validation"]:
        proc_dataset[split] = []
        for i in range(len(my_dataset[split])):
            proc_dataset[split].append(
                convert_causal(
                    my_tokenizer,
                    PROMPT.format(message=my_dataset[split][i]["question"]),
                    my_dataset[split][i]["answerText"],
                )
            )
    return proc_dataset


def generate(data, my_model, my_tokenizer, num_beams, my_device):
    """generate the answer candidates"""
    data = my_tokenizer(data, return_tensors="pt")["input_ids"].to(my_device)
    output_ids = my_model.generate(
        data,
        do_sample=False,
        num_beams=num_beams,
        num_beam_groups=num_beams // 10,
        num_return_sequences=num_beams,
        max_new_tokens=128,
        diversity_penalty=0.2,
        eos_token_id=my_tokenizer.eos_token_id,
        pad_token_id=my_tokenizer.eos_token_id,
    )
    output_ids = [j[data.shape[1] :] for j in output_ids]
    output = my_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return output


def train(args, dataset):
    """train/finetune the pretrained decoder only llm"""
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    processed_data = process_data(dataset, tokenizer)

    training_args = TrainingArguments(
        num_train_epochs=3,
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=3,
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=16,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        weight_decay=0.05,
        report_to="wandb",
        seed=8,
        bf16=True,
    )

    training_args.set_dataloader(pin_memory=False)
    model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(5, early_stopping_threshold=1e-4)],
    )

    trainer.train()
    model.save_pretrained(Path(args.output_dir) / "checkpoint-best")


def evaluate(args, dataset):
    """eval & generate answer candidates using the trained llm"""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        model, get_best_checkpoint_path(args.output_dir), torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    eval_split = args.evaluation_split
    prompts = create_prompts(dataset, eval_split)
    with open(
        Path(args.output_dir) / f"{args.model_name}_{eval_split}_answer_candidates",
        "rb",
    ) as file:
        mistral_answer = pickle.load(file)

    # filling the pkl file with the generated answers
    for index in tqdm(range(len(mistral_answer), len(dataset[eval_split]))):
        mistral_answer += [
            {
                "id": dataset[eval_split][index]["id"],
                "question": dataset[eval_split][index]["question"],
                "true_answer": dataset[eval_split][index]["answerText"],
                "model_answer": generate(
                    prompts[index], model, tokenizer, args.num_beams, device
                ),
            }
        ]

        with open(args.file_name, "wb") as file:
            pickle.dump(mistral_answer, file)


if __name__ == "__main__":
    huggingface_hub.login()
    args = parser.parse_args()

    ds = datasets.load_dataset("AmazonScience/mintaka")

    if args.mode == "train":
        train(args, ds)
    elif args.mode == "eval":
        evaluate(args, ds)
    elif args.mode == "train_eval":
        train(args, ds)
        evaluate(args, ds)
    else:
        raise ValueError(
            f"Wrong mode argument passed: must be train or eval, passed {args.mode}"
        )
