import json
import logging
import pandas as pd
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import datasets
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from ..config import SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
from ..utils import entities_to_labels
from ..wikidata import WikidataEntityToLabel, WikidataRedirectsCache


def load_model_and_tokenizer_by_name(
    model_name: str, from_pretrained_path: str = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """load_model_and_tokenizer_by_name - helper for loading model and tokenizer by name

    Args:
        model_name (str): name of HF model: SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
        from_pretrained_path (str, optional): If provided, will load from local, else from hub. Defaults to None.

    Raises:
        ValueError: Exception, if passed wrong model_name

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and Tokenizer
    """

    if from_pretrained_path is not None:
        logging.info(f"Load local checkpoint model from {from_pretrained_path}")
        model_path = Path(from_pretrained_path)
    else:
        logging.info(f"No checkpint, load public pretrained model {model_name}")
        model_path = model_name

    if model_name in SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        raise ValueError(
            f"model_name must be one of {SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES}, but passed {model_name}"
        )

    return model, tokenizer


def augmentation_by_redirects(
    example_batch: Dict, wikidata_redirects: WikidataRedirectsCache
):
    """augmentation_by_redirects function for HF dataset for applying augmentation
    by adding all possible redirects for labels

    Args:
        example_batch (Dict): HF Dataset batch
        wikidata_redirects (WikidataRedirectsCache): Service for extracting redirects
    """
    outputs = {
        "subject": [],
        "property": [],
        "object": [],
        "question": [],
    }
    for idx in range(len(example_batch["object"])):
        outputs["subject"].append(example_batch["subject"][idx])
        outputs["property"].append(example_batch["property"][idx])
        outputs["object"].append(example_batch["object"][idx])
        outputs["question"].append(example_batch["question"][idx])

        for redirect in wikidata_redirects.get_redirects(example_batch["object"][idx]):
            outputs["subject"].append(example_batch["subject"][idx])
            outputs["property"].append(example_batch["property"][idx])
            outputs["object"].append(redirect)
            outputs["question"].append(example_batch["question"][idx])

    return outputs


def convert_to_features(
    example_batch: Dict,
    tokenizer: PreTrainedTokenizer,
    question_feature_name: str = "question",
    label_feature_name: str = "object",
) -> Dict:
    """convert_to_features function for HF dataset for applying tokenizer

    Args:
        example_batch (Dict): HF Dataset batch
        tokenizer (PreTrainedTokenizer): HF Tokenizer
        question_feature_name (str): Name of column with quesions
        label_feature_name (str): Name of column with labels

    Returns:
        Dict: HF Dataset tokenized batch
    """
    input_encodings = tokenizer(
        example_batch[question_feature_name],
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    target_encodings = tokenizer(
        [
            obj[0] if isinstance(obj, list) else obj
            for obj in example_batch[label_feature_name]
        ],
        padding="max_length",
        truncation=True,
        max_length=64,
    )

    labels = target_encodings["input_ids"]

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": labels,
    }

    return encodings


def _preprocessing_kbqa_dataset(example, entity2label: WikidataEntityToLabel):
    example["object"] = entities_to_labels(example["object"], entity2label)
    return example


def _filter_objects(example):
    if isinstance(example["object"], str):
        return True
    elif isinstance(example["object"], list):
        return all(isinstance(obj, str) for obj in example["object"])
    else:
        return False


def load_kbqa_seq2seq_dataset(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer: PreTrainedTokenizer,
    dataset_cache_dir: str = None,
    split: str = None,
    apply_redirects_augmentation: bool = False,
    entity2label: WikidataEntityToLabel = WikidataEntityToLabel(),
    use_convert_to_features: bool = True,
) -> datasets.arrow_dataset.Dataset:
    """load_kbqa_seq2seq_dataset - helper for load dataset for seq2seq KBQA

    Args:
        dataset_name (str): name or path to HF dataset with str fields: object, question
        dataset_config_name (str): HF dataset config name
        tokenizer (PreTrainedTokenizer): Tokenizer of seq2seq model
        dataset_cache_dir (str, optional): Path to HF cache. Defaults to None.
        split (str, optional):
            Load only train/validation/test split of dataset if passed, else load all.
            Defaults to None
        apply_redirects_augmentation (bool, optional): Using wikidata redirect for augmention,
            Defaults to False
        entity2label (WikidataEntityToLabel, optional): Used for converting entities to label if provided IDs
            Defaults to WikidataEntityToLabel()
        use_convert_to_features (bool, optional): Converting dataset to features for seq2seq traning/evalutaion pipeline
            Defaults to True

    Returns:
        datasets.arrow_dataset.Dataset: Prepared dataset for seq2seq
    """

    dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=dataset_cache_dir,
        verification_mode="no_checks",
        split=split,
    )
    dataset = dataset.map(
        lambda example: _preprocessing_kbqa_dataset(example, entity2label)
    )
    dataset = dataset.filter(_filter_objects)
    if apply_redirects_augmentation:
        wikidata_redirects = WikidataRedirectsCache()
        dataset["train"] = dataset["train"].map(
            lambda batch: augmentation_by_redirects(batch, wikidata_redirects),
            batched=True,
        )

    if use_convert_to_features is True:
        dataset = dataset.map(
            lambda batch: convert_to_features(batch, tokenizer),
            batched=True,
        )
        columns = [
            "input_ids",
            "labels",
            "attention_mask",
        ]
        dataset.set_format(type="torch", columns=columns)

    return dataset


def load_lcquad2_seq2seq_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    dataset_cache_dir: str = None,
    split: str = "train",
    use_convert_to_features: bool = True,
) -> datasets.arrow_dataset.Dataset:
    """load_lcquad2_seq2seq_dataset - helper for loading dataset for seq2seq lcquad2

    Args:
        dataset_name (str): Hugging Face dataset name
        tokenizer (PreTrainedTokenizer): Tokenizer of seq2seq model
        dataset_cache_dir (str, optional): Path to HF cache. Defaults to None.
        split (str, optional): Dataset split to load ("train" or "test"). Defaults to "train".
        use_convert_to_features (bool, optional): Convert dataset to features for seq2seq training/evaluation pipeline.
        Defaults to True.

    Returns:
        datasets.arrow_dataset.Dataset: Prepared dataset for seq2seq
    """

    dataset = datasets.load_dataset(
        dataset_name,
        cache_dir=dataset_cache_dir,
        split=split,
    )

    def preprocess_function(examples):
        inputs = examples["Question"]
        labels = examples["Label"]
        inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=64)
        labels = tokenizer(labels, truncation=True, padding="max_length", max_length=64)
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
        }

    dataset = dataset.map(preprocess_function, batched=True)

    if use_convert_to_features:
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    return dataset


def load_mintaka_seq2seq_dataset(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = None,
    use_convert_to_features: bool = True,
):
    dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        verification_mode="no_checks",
        split=split,
    )

    if use_convert_to_features is True:
        dataset = dataset.map(
            lambda batch: convert_to_features(
                batch, tokenizer, label_feature_name="answerText"
            ),
            batched=True,
        )
        columns = [
            "input_ids",
            "labels",
            "attention_mask",
        ]
        dataset.set_format(type="torch", columns=columns)

    return dataset


def hf_model_name_mormolize(model_name: str) -> str:
    """hf_model_name_mormolize - return normolized model name for storing to directory
    Example: facebook/bart-large -> facebook_bart-large

    Args:
        model_name (str): HF model_name

    Returns:
        str: normolized model_name
    """
    return model_name.replace("/", "_")


def get_model_logging_dirs(save_dir, model_name, run_name=None):
    normolized_model_name = hf_model_name_mormolize(model_name)

    run_path = Path(save_dir)
    if run_name is not None:
        run_path = run_path / run_name
    run_path = run_path / normolized_model_name

    model_dir = run_path / "models"
    logging_dir = run_path / "logs"

    return model_dir, logging_dir, normolized_model_name


def dump_eval(
    results_df: pd.DataFrame, report: dict, args: Namespace, normolized_model_name: str
):
    eval_report_dir = Path(args.save_dir)
    if args.run_name is not None:
        eval_report_dir = eval_report_dir / args.run_name
    eval_report_dir = (
        eval_report_dir
        / (
            normolized_model_name.name
            if isinstance(normolized_model_name, Path)
            else str(normolized_model_name)
        )
        / "evaluation"
    )

    number_of_versions = len(list(eval_report_dir.glob("version_*")))
    eval_report_dir = eval_report_dir / f"version_{number_of_versions}"
    eval_report_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(eval_report_dir / "results.csv", index=False)
    with open(eval_report_dir / "report.json", "w", encoding=None) as file_handler:
        json.dump(report, file_handler)
    with open(eval_report_dir / "args.json", "w", encoding=None) as file_handler:
        json.dump(vars(args), file_handler)

    return eval_report_dir
