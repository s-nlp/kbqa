from __future__ import annotations

import json

import datasets
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def load_dataset(dataset_path: dict[str, str]) -> DatasetDict:
    if "huggingface" in dataset_path:
        return datasets.load_dataset(dataset_path["huggingface"])
    elif "webnlg" in dataset_path:
        with open(dataset_path["webnlg"]["train"], "r") as f:
            train_ds = json.load(f)
        with open(dataset_path["webnlg"]["valid"], "r") as f:
            valid_ds = json.load(f)
        with open(dataset_path["webnlg"]["test"], "r") as f:
            test_ds = json.load(f)
        return DatasetDict(
            {
                "train": train_ds,
                "valid": valid_ds,
                "test": test_ds,
            }
        )
    else:
        train_ds = Dataset.from_parquet(dataset_path["train"])
        valid_ds = Dataset.from_parquet(dataset_path["valid"])
        test_ds = Dataset.from_parquet(dataset_path["test"])
        return DatasetDict(
            {
                "train": train_ds,
                "valid": valid_ds,
                "test": test_ds,
            }
        )


def tokenize_function(
    batch: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 1024,
) -> dict:
    """tokenize_function method for tokenize data

    Parameters
    ----------
    batch : dict
        batch of data from dataset
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
        Tokenizer used for tokenize texts
    max_length : int, optional
        Max length of data, by default 1024

    Returns
    -------
    dict
        Dict with input_ids, attention_mask and labels after tokenization
    """
    tokenized_question = tokenizer(
        batch["Question"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenized_answer = tokenizer(
        batch["Answer"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": tokenized_question["input_ids"],
        "attention_mask": tokenized_question["attention_mask"],
        "labels": tokenized_answer["input_ids"],
    }


def prepare_dataset(
    ds: Dataset | DatasetDict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 1024,
    num_proc: int = 1,
) -> Dataset | DatasetDict:
    """prepare_dataset Tokenize and convert data to PyTorch format

    Parameters
    ----------
    ds : Dataset | DatasetDict
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
    max_length : int, optional
         Max length of data, by default 1024
    num_proc : int, optional
        Number of processes used by tokenizer, by default 1

    Returns
    -------
    Dataset | DatasetDict
        Preprocessed dataset
    """
    ds = ds.map(
        lambda batch: tokenize_function(
            batch,
            tokenizer,
            max_length,
        ),
        batched=True,
        num_proc=num_proc,
    )

    columns = [
        "input_ids",
        "labels",
        "attention_mask",
    ]
    ds.set_format(type="torch", columns=columns)
    return ds


def get_preprocessed_dataset(
    dataset,
    tokenizer,
    question_column,
    answer_column,
    max_seq_length,
    max_answer_length,
    padding,
    preprocessing_num_workers,
    ignore_pad_token_for_loss=True,
    ignore_pad_token=-100,
):
    def preprocess_function(examples):
        inputs, targets = examples[question_column], examples[answer_column]

        model_inputs = tokenizer(
            inputs, max_length=max_seq_length, padding=padding, truncation=True
        )
        # Tokenize targets with text_target=...
        labels = tokenizer(
            text_target=targets,
            max_length=max_answer_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id
        # in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (lbl if lbl != tokenizer.pad_token_id else ignore_pad_token)
                    for lbl in label
                ]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    return dataset
