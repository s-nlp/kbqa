# pyling: disable=no-member

from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import logging
import datasets
import torch
import pandas as pd
from torch.utils.data import DataLoader


def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    batch_size: int = 2,
    num_beams: int = 5,
    device: torch.device = torch.device("cuda"),
) -> List[str]:
    """predict results for HF dataset. Pass needed split.

    Args:
        model (PreTrainedModel): model for predicting
        tokenizer (PreTrainedTokenizer): tokenizer for decoding predicted results
        dataset (datasets.arrow_dataset.Dataset): HF dataset
        batch_size (int, optional): Batch size. Defaults to 2.
        num_beams (int, optional): Num beans. Defaults to 5.
        device (torch.device, optional): torch device: cuda or cpu. Defaults to torch.device('cuda').

    Returns:
        List[str]: List of predicted and decoded results
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)

    generated_decoded = []
    for batch in logging.tqdm(dataloader, "evaluate model"):
        generated_ids = model.generate(
            batch["input_ids"].to(device), num_beams=num_beams
        )
        generated_decoded.extend(
            tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    return generated_decoded


def make_report(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    device: torch.device = torch.device("cuda"),
) -> Tuple[pd.DataFrame, dict]:
    """make_report

    Args:
        model (PreTrainedModel): model for predicting
        tokenizer (PreTrainedTokenizer): tokenizer for decoding predicted results
        dataset (datasets.arrow_dataset.Dataset): HF dataset

    Returns:
        Tuple[pd.DataFrame, dict]: results of predicting with questions on DataFrame and report dict with metrics
    """
    preds = predict(model, tokenizer, dataset, device=device)
    results_df = pd.DataFrame(
        {"question": dataset["question"], "answer": preds, "target": dataset["object"]}
    )
    vocab = tokenizer.get_vocab()
    results_df["target_out_of_vocab"] = results_df["target"].apply(
        lambda lbl: vocab["<unk>"] in tokenizer.encode(lbl)
    )

    report = {
        "num_true_positive": results_df[
            results_df["answer"] == results_df["target"]
        ].index.size,
        "num_total": results_df.index.size,
    }
    report["accuracy"] = report["num_true_positive"] / report["num_total"]
    report["num_out_of_vocab"] = results_df[
        results_df["target_out_of_vocab"]
    ].index.size

    return results_df, report
