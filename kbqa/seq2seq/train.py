import datasets
from .redirect_trainer import Seq2SeqWikidataRedirectsTrainer
from ..wikidata.wikidata_redirects import WikidataRedirectsCache


from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def train(
    run_name: str,
    report_to: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: datasets.Dataset,
    valid_dataset: datasets.Dataset,
    output_dir: str = "./runs/models/model",
    logging_dir: str = "./runs/logs/model",
    save_total_limit: int = 2,
    num_train_epochs: int = 8,
    max_steps: int = -1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    logging_steps: int = 500,
    gradient_accumulation_steps: int = 8,
    trainer_mode: str = "default",
) -> Seq2SeqTrainer:
    """train seq2seq model for KBQA problem
    Work with HF dataset with object and question field (str)

    Args:
        run_name: (str): Run name
        report_to: (str): The list of integrations to report the results and logs to.
            Supported platforms are "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard","clearml" and "wandb".
            Use "all" to report to all integrations installed, "none" for no integrations.
        model: (PreTrainedModel): HF Model for training
        tokenizer: (PreTrainedTokenizer): HF Tokenizer for training (provided with model usually)
        train_dataset (datasets.Dataset): HF Dataset object for traning
        valid_dataset (datasets.Dataset): HF Dataset object for validation
        output_dir (str): Path to directory for storing model's checkpoints . Defaults to './runs/models/model'
        logging_dir (str): Path to directory for storing traning logs . Defaults to './runs/logs/model'
        save_total_limit (int, optional): Total limit for storing model's checkpoints. Defaults to 5.
        num_train_epochs (int, optional): Total number of traning epoches. Defaults to 8.
        max_steps (int, optional, defaults to -1): If set to a positive number,
            the total number of training steps to perform. Overrides num_train_epochs.
            In case of using a finite iterable dataset the training may stop before reaching
            the set number of steps when all data is exhausted
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
        Seq2SeqTrainer: Trained after traning and validation
    """
    training_args = Seq2SeqTrainingArguments(
        run_name=run_name,
        report_to=report_to,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    if trainer_mode == "default":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            # compute_metrics=lambda x: compute_metrics(x, tokenizer, False)
        )
    elif trainer_mode == "Seq2SeqWikidataRedirectsTrainer":
        redirect_cache = WikidataRedirectsCache()

        trainer = Seq2SeqWikidataRedirectsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            redirect_cache=redirect_cache,
            # compute_metrics=lambda x: compute_metrics(x, tokenizer=tokenizer),
        )
    else:
        raise ValueError(
            'trainer_mode must be "default" or "Seq2SeqWikidataRedirectsTrainer", '
            f"but provided {trainer_mode}"
        )

    trainer.train()

    return trainer
