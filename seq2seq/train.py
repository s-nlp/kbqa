from typing import Tuple, Optional
import datasets
from seq2seq.utils import load_model_and_tokenizer_by_name, load_kbqa_seq2seq_dataset
from seq2seq.redirect_trainer import Seq2SeqWikidataRedirectsTrainer

from transformers import (
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def train(
    model_name: str,
    dataset_name: str,
    dataset_config_name: str,
    dataset_cache_dir: Optional[str] = None,
    output_dir: str = "./runs/models/model",
    logging_dir: str = "./runs/logs/model",
    save_total_limit: int = 5,
    num_train_epochs: int = 8,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    logging_steps: int = 500,
    gradient_accumulation_steps: int = 8,
) -> Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]:
    """train seq2seq model for KBQA problem
    Work with HF dataset with object and question field (str)

    Args:
        model_name (str): HF seq2seq model: SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
        dataset_name (str): name or path to HF dataset with str fields: object, question
        dataset_config_name (str): HF dataset config name
        dataset_cache_dir (str, optional): Path to HF cache. Defaults to None.
        output_dir (str): Path to directory for storing model's checkpoints . Defaults to './runs/models/model'
        logging_dir (str): Path to directory for storing traning logs . Defaults to './runs/logs/model'
        save_total_limit (int, optional): Total limit for storing model's checkpoints. Defaults to 5.
        num_train_epochs (int, optional): Total number of traning epoches. Defaults to 8.
        per_device_train_batch_size (int, optional): train batch size per device. Defaults to 1.
        per_device_eval_batch_size (int, optional): eval batch size per device. Defaults to 1.
        warmup_steps (int, optional): warmup steps for traning. Defaults to 500.
        weight_decay (float, optional): weight decay for traning. Defaults to 0.01.
        evaluation_strategy (str, optional):
            "no": No evaluation is done during training;
            "steps": Evaluation is done (and logged) every eval_steps;
            "epoch": Evaluation is done at the end of each epoch;
            Defaults to 'steps'.
        eval_steps (int, optional):
            Number of update steps between two evaluations if evaluation_strategy="steps".
            Will default to the same value as logging_steps if not set.
            Defaults to 500.
        logging_steps (int, optional):
            Number of update steps between two logs if logging_strategy="steps".
            Defaults to 500.
        gradient_accumulation_steps (int, optional):
             Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
             Defaults to 8.

    Returns:
        Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]: _description_
    """
    model, tokenizer = load_model_and_tokenizer_by_name(model_name)

    dataset = load_kbqa_seq2seq_dataset(
        dataset_name, dataset_config_name, tokenizer, dataset_cache_dir
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Seq2SeqWikidataRedirectsTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()

    return trainer, model, dataset
