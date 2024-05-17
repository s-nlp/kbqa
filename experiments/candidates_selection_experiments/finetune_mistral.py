import os
import torch
import pickle
import random
import datasets
import huggingface_hub

import numpy as np
import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)

MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
OUTPUT_DIR = './mixtral_mintaka_checkpoints_4bit'

seed = 13
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

huggingface_hub.login()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def convert_causal(tokenizer, source, target, max_length=256):
    source_tokens = tokenizer(
        source,
        add_special_tokens=False,
        max_length=max_length,
        padding=False,
        truncation=True
    )["input_ids"]
    if tokenizer.bos_token_id:
        source_tokens.insert(0, tokenizer.bos_token_id)
    input_ids = source_tokens[:]
    target_tokens = tokenizer(
        target,
        add_special_tokens=False,
        max_length=max_length,
        padding=False,
        truncation=True
    )["input_ids"]
    input_ids += target_tokens + [tokenizer.eos_token_id]
    input_ids = torch.LongTensor(input_ids)
    labels = input_ids.clone()
    attention_mask = input_ids.new_ones(input_ids.size())
    labels[:len(source_tokens)] = -100
    assert input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= 2 * max_length + 2

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

prompt = '[INST] Answer as briefly as possible without additional information.\n{message} [/INST]'
dataset = datasets.load_dataset('AmazonScience/mintaka')
tokenized_dataset = {'train': [], 'test': [], 'validation': []}
for split in ['train', 'test', 'validation']:
    for i in range(len(dataset[split])):
        tokenized_dataset[split].append(convert_causal(
            tokenizer, 
            prompt.format(message=dataset[split][i]['question']),
            dataset[split][i]['answerText']
        ))
train_dataset, eval_dataset = tokenized_dataset['train'], tokenized_dataset['validation']

training_args = TrainingArguments(
    num_train_epochs=3,
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=3,
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
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
    bf16=True
)

training_args.set_dataloader(pin_memory=False)
model.config.use_cache = False

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Trainer(model=model, args=training_args, 
                  train_dataset=train_dataset, eval_dataset=eval_dataset, 
                  data_collator=data_collator, callbacks=[EarlyStoppingCallback(5, early_stopping_threshold=1e-4)])

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, 'final_checkpoint/'))
