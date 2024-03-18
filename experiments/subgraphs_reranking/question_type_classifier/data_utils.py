"""data utils for question type classifier"""
import torch


def convert_to_features(
    complexity_type_to_id,
    example_batch,
    tokenizer,
    question_feature_name: str = "question",
):
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
        max_length=128,
    )

    labels = []
    for label in example_batch["complexityType"]:
        labels.append(complexity_type_to_id.get(label, complexity_type_to_id["other"]))
    labels = torch.LongTensor(labels)  # pylint: disable=no-member

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": labels,
    }

    return encodings
