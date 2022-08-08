from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PreTrainedTokenizer, PreTrainedModel
import datasets
from typing import Dict, Tuple
from pathlib import Path
from config import SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES


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
        model_path = Path(from_pretrained_path)
    else:
        model_path = model_name

    if model_name in ["facebook/bart-base", "facebook/bart-large"]:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    elif model_name in ["t5-small", "t5-base", "t5-large"]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        raise ValueError(
            f"model_name must be one of {SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES}, but passed {model_name}"
        )

    return model, tokenizer


def convert_to_features(example_batch: Dict, tokenizer: PreTrainedTokenizer) -> Dict:
    """convert_to_features function for HF dataset for applying tokenizer

    Args:
        example_batch (Dict): HF Dataset batch
        tokenizer (PreTrainedTokenizer): HF Tokenizer

    Returns:
        Dict: HF Dataset tokenized batch
    """
    input_encodings = tokenizer.batch_encode_plus(
        example_batch["question"],
        pad_to_max_length=True,
        max_length=1024,
        truncation=True,
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["object"],
        pad_to_max_length=True,
        max_length=1024,
        truncation=True,
    )

    labels = target_encodings["input_ids"]

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": labels,
    }

    return encodings


def load_kbqa_seq2seq_dataset(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer: PreTrainedTokenizer,
    dataset_cache_dir: str = None,
    split: str = None,
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

    Returns:
        datasets.arrow_dataset.Dataset: Prepared dataset for seq2seq
    """

    dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=dataset_cache_dir,
        ignore_verifications=True,
        split=split,
    )

    dataset = dataset.filter(lambda example: isinstance(example["object"], str))
    # dataset = dataset.filter(lambda example, idx: idx % 500 == 0, with_indices=True)
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


def hf_model_name_mormolize(model_name: str) -> str:
    """hf_model_name_mormolize - return normolized model name for storing to directory
    Example: facebook/bart-large -> facebook_bart-large

    Args:
        model_name (str): HF model_name

    Returns:
        str: normolized model_name
    """
    return model_name.replace("/", "_")
