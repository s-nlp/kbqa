"""Train model for ranking subgraphs.
MSE Loss
"""
import argparse
import os
import random
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_utils import SequenceDataset, data_df_convert


torch.manual_seed(8)
random.seed(8)
np.random.seed(8)


METRIC_CLASSIFIER = evaluate.combine(
    [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "hyperml/balanced_accuracy",
    ]
)
METRIC_REGRESSION = evaluate.combine(["mae"])


parse = argparse.ArgumentParser()
parse.add_argument(
    "run_name",
    type=str,
    help="run_name - name for folder that will be created in output_path for storing model. Also used for wandb",
)
parse.add_argument(
    "train_data_path",
    type=str,
    help="Path to train JSONL file",
)

parse.add_argument(
    "valid_data_path",
    type=str,
    help="Path to valid JSONL file",
)

parse.add_argument(
    "test_data_path",
    type=str,
    help="Path to test JSONL file",
)

parse.add_argument(
    "--output_path",
    type=str,
    default="/mnt/storage/QA_System_Project/subgrraphs_reranking_runs/",
)

parse.add_argument(
    "--fit_on_train_and_val",
    default=True,
    type=lambda x: (str(x).lower() == "true"),
    help="If True, train and valid data will be used for fit model. Default True.",
)

parse.add_argument(
    "--model_name",
    type=str,
    default="sentence-transformers/all-mpnet-base-v2",
    help="HF model name for AutoModelForSequenceClassification",
)

parse.add_argument(
    "--classification_threshold",
    type=float,
    default=0.5,
    help="Threashold for classification prediction. From 0 to 1, Default 0.5",
)

parse.add_argument(
    "--wandb_on",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="Using WanDB or not (True/False)",
)

parse.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=32,
)

parse.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=64,
)

parse.add_argument(
    "--num_train_epochs",
    type=int,
    default=6,
)
parse.add_argument(
    "--do_highlighting",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="True/False. If True, add highliting tokens for candidate in linearized graph",
)
parse.add_argument(
    "--do_linearization",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help='True/False. If False, "question + [SEP] + candiadte label" will used as a input for ranker model',
)


def create_sampler(target):
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


def compute_metrics(eval_pred, classification_threshold):
    predictions, labels = eval_pred
    results = METRIC_REGRESSION.compute(predictions=predictions, references=labels)

    predictions = predictions > classification_threshold
    results.update(
        METRIC_CLASSIFIER.compute(predictions=predictions, references=labels)
    )
    return results


def main(args):
    train_df = pd.read_json(args.train_data_path, lines=True)
    valid_df = pd.read_json(args.valid_data_path, lines=True)
    test_df = pd.read_json(args.test_data_path, lines=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
    )

    if args.do_highlighting:
        candidate_start_token = "[unused1]"
        candidate_end_token = "[unused2]"
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[unused1]", "[unused2]"]}
        )
    else:
        candidate_start_token = ""
        candidate_end_token = ""

    train_df = data_df_convert(
        train_df,
        sep_token=tokenizer.sep_token,
        candidate_start_token=candidate_start_token,
        candidate_end_token=candidate_end_token,
        linearization=args.do_linearization,
    )
    valid_df = data_df_convert(
        valid_df,
        sep_token=tokenizer.sep_token,
        candidate_start_token=candidate_start_token,
        candidate_end_token=candidate_end_token,
        linearization=args.do_linearization,
    )
    test_df = data_df_convert(
        test_df,
        sep_token=tokenizer.sep_token,
        candidate_start_token=candidate_start_token,
        candidate_end_token=candidate_end_token,
        linearization=args.do_linearization,
    )

    if args.fit_on_train_and_val:
        train_df = pd.concat([train_df, valid_df])

    train_dataset = SequenceDataset(train_df, tokenizer)
    test_dataset = SequenceDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=Path(args.output_path) / args.run_name / "outputs",
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=Path(args.output_path) / args.run_name / "logs",
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        report_to="wandb" if args.wandb_on else "tensorboard",
    )

    class CustomTrainer(Trainer):
        def get_train_dataloader(self) -> torch.utils.data.DataLoader:
            train_sampler = create_sampler(train_df["correct"].astype(int).ravel())
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=32, sampler=train_sampler
            )
            return train_loader

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda x: compute_metrics(x, args.classification_threshold),
    )
    trainer.train()

    checkpoint_best_path = (
        Path(args.output_path) / args.run_name / "outputs" / "checkpoint-best"
    )
    model.save_pretrained(checkpoint_best_path)
    tokenizer.save_pretrained(checkpoint_best_path)

    print("Model dumbed to ", checkpoint_best_path)

    print("\nLast one evaluation:\n\n", trainer.evaluate())


if __name__ == "__main__":
    args = parse.parse_args()

    if args.wandb_on:
        os.environ["WANDB_NAME"] = args.run_name

    if args.do_linearization is False and args.do_highlighting is True:
        raise ValueError(
            "It is not possible to use do_highlighting and not to use do_linearization"
        )

    main(args)
