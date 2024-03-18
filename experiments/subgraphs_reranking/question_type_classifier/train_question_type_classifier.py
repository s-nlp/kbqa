"""Train model for ranking subgraphs.
MSE Loss
"""
import argparse
import os
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
from data_utils import convert_to_features

torch.manual_seed(8)
random.seed(8)
np.random.seed(8)


METRIC_CLASSIFIER = evaluate.combine(["hyperml/balanced_accuracy"])


parse = argparse.ArgumentParser()
parse.add_argument(
    "--run_name",
    default="tmp",
    type=str,
    help="folder name inside output_path for storing model. Also used for wandb",
)
parse.add_argument(
    "--data_path",
    type=str,
    default="AmazonScience/mintaka",
    help="Path to train JSONL file",
)

parse.add_argument(
    "--output_path",
    type=str,
    default="./question_types_classifier_runs/",
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
    default=16,
)

parse.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=32,
)

parse.add_argument(
    "--num_train_epochs",
    type=int,
    default=6,
)


def compute_metrics(eval_pred):
    """custom metrics for training"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return METRIC_CLASSIFIER.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    """custom trainer with sampler"""

    def get_labels(self):
        """get labels from train dataset"""
        labels = []
        for i in self.train_dataset:
            labels.append(int(i["labels"].cpu().detach().numpy()))
        return labels

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        """create our custom sampler"""
        labels = self.get_labels()
        return self.create_sampler(labels)

    def create_sampler(self, target):
        """ "weighted random sampler"""
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)  # pylint: disable=no-member
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


if __name__ == "__main__":
    args = parse.parse_args()

    if args.wandb_on:
        os.environ["WANDB_NAME"] = args.run_name
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    complexity_type_to_id = {
        "count": 0,
        "yesno": 1,
        "other": 2,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(complexity_type_to_id),
    )

    # getting mintaka and mapping complexity type to id
    mintaka_dataset = load_dataset(args.data_path, "en")
    mintaka_dataset = mintaka_dataset.map(
        lambda batch: convert_to_features(complexity_type_to_id, batch, tokenizer),
        batched=True,
    )
    columns = ["input_ids", "labels", "attention_mask"]
    mintaka_dataset.set_format(type="torch", columns=columns)

    training_args = TrainingArguments(
        output_dir=Path(args.output_path) / args.run_name / "outputs",
        save_total_limit=1,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        logging_dir=Path(args.output_path) / args.run_name / "logs",
        greater_is_better=True,
        logging_steps=250,
        save_steps=250,
        evaluation_strategy="steps",
        report_to="wandb",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=mintaka_dataset["train"],
        eval_dataset=mintaka_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    checkpoint_best_path = (
        Path(args.output_path) / args.run_name / "outputs" / "checkpoint-best"
    )
    model.save_pretrained(checkpoint_best_path)
    tokenizer.save_pretrained(checkpoint_best_path)

    print("Model dumped to ", checkpoint_best_path)
    print("\nFinal evaluation:\n\n", trainer.evaluate())
