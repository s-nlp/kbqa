from typing import Tuple, Optional
import datasets
from seq2seq.utils import load_model_and_tokenizer_by_name, load_kbqa_seq2seq_dataset
from seq2seq.redirect_trainer import Seq2SeqWikidataRedirectsTrainer
from caches.wikidata_redirects import WikidataRedirectsCache
from seq2seq.eval import compute_metrics

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
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
    trainer_mode: str = 'default',
) -> Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]:
    """train seq2seq model for KBQA problem
    Work with HF dataset with object and question field (str)

    Args:
        model: (PreTrainedModel): HF Model for training
        tokenizer: (PreTrainedTokenizer): HF Tokenizer for training (provided with model usually)
        dataset (datasets.Dataset): HF Dataset object with 'train' and 'validation'
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
        trainer_mode (str, optional):
            trainer mode, as a default will used Seq2SeqTrainer, but if provided  
            Seq2SeqWikidataRedirectsTrainer, that it will used.
            Default to 'default'

    Returns:
        Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]: _description_
    """
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

    redirect_cache = WikidataRedirectsCache()

    if trainer_mode == 'default':
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
        )
    elif trainer_mode == 'Seq2SeqWikidataRedirectsTrainer':
        trainer = Seq2SeqWikidataRedirectsTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            redirect_cache=redirect_cache,
            compute_metrics=compute_metrics,
        )
    else:
        raise ValueError(
            'trainer_mode must be "default" or "Seq2SeqWikidataRedirectsTrainer", '
            f'but provided {trainer_mode}'
        )

    trainer.train()

    return trainer, model, dataset
