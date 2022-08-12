# pyling: disable=no-member

from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import logging
import datasets
import torch
import pandas as pd
from torch.utils.data import DataLoader
from caches.wikidata_redirects import WikidataRedirectsCache
import evaluate


def predict_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    batch_size: int = 2,
    num_beams: int = 10,
    num_return_sequences: int = 10,
    device: torch.device = torch.device("cuda"),
) -> List[str]:
    """predict answers results for HF dataset. Pass needed split.

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

    generated_decoded = {f"answer_{idx}": [] for idx in range(num_return_sequences)}
    for batch in logging.tqdm(dataloader, "evaluate model"):
        generated_ids = model.generate(
            batch["input_ids"].to(device),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        generated_decoded_batch = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

        current_batch_size = batch["input_ids"].shape[0]
        for answer_idx, start_batch_idx in enumerate(
            range(0, current_batch_size * num_return_sequences, current_batch_size)
        ):
            generated_decoded[f"answer_{answer_idx}"].extend(
                generated_decoded_batch[
                    start_batch_idx : start_batch_idx + current_batch_size
                ]
            )

    return generated_decoded


def _get_accuracy_for_report(report: dict, results_df: pd.DataFrame, top_k: int = 0):
    report[f"num_true_positive_top{top_k}"] = results_df[
        results_df[f"answer_{top_k}"] == results_df["target"]
    ].index.size
    report["num_total"] = results_df.index.size
    report[f"accuracy_top{top_k}"] = (
        report[f"num_true_positive_top{top_k}"] / report["num_total"]
    )
    return report


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decoding the predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    redirect_labels = []
    for label in decoded_labels:

        # getting the redirect for the current label
        redirects = WikidataRedirectsCache.get_redirects(label)
        redirect_labels.append[redirects]

    # calculating bleu score (list('str') vs list(list('str')))
    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=redirect_labels)
    return {"bleu": result["score"]}


def make_report(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    batch_size: int,
    num_return_sequences: int = 10,
    device: torch.device = torch.device("cuda"),
) -> Tuple[pd.DataFrame, dict]:
    """make_report

    Args:
        model (PreTrainedModel): model for predicting
        tokenizer (PreTrainedTokenizer): tokenizer for decoding predicted results
        dataset (datasets.arrow_dataset.Dataset): HF dataset
        batch_size (int): batch size for evaluation
        num_return_sequences(int): Number generated answers from beam search process

    Returns:
        Tuple[pd.DataFrame, dict]: results of predicting with questions on DataFrame and report dict with metrics
    """

    results_df = pd.DataFrame(
        {"question": dataset["question"], "target": dataset["object"]}
    )
    generated_answers = predict_answers(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_return_sequences=num_return_sequences,
        batch_size=batch_size,
        device=device,
    )
    for key, vals in generated_answers.items():
        results_df[key] = vals

    vocab = tokenizer.get_vocab()
    results_df["target_out_of_vocab"] = results_df["target"].apply(
        lambda lbl: vocab["<unk>"] in tokenizer.encode(lbl)
    )
    report = {}
    report["num_out_of_vocab"] = results_df[
        results_df["target_out_of_vocab"]
    ].index.size
    for top_k in range(len(generated_answers.keys())):
        report = _get_accuracy_for_report(report, results_df, top_k)

    return results_df, report
