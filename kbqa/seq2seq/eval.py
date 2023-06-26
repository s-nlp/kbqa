# pyling: disable=no-member

from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import logging
import datasets
import torch
import pandas as pd
from torch.utils.data import DataLoader
from ..wikidata.wikidata_redirects import WikidataRedirectsCache
from ..metrics.recall import recall
from ..utils import get_default_logger

try:
    import evaluate
except:
    pass

logger = get_default_logger()


def predict_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    batch_size: int = 2,
    num_beams: int = 500,
    num_return_sequences: int = 500,
    num_beam_groups: int = 50,
    diversity_penalty: int = 0.1,
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
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )
        generated_decoded_batch = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        current_batch_size = batch["input_ids"].shape[0]
        for start_batch_idx in range(
            0, current_batch_size * num_return_sequences, num_return_sequences
        ):
            for answer_idx, answer in enumerate(
                generated_decoded_batch[
                    start_batch_idx : start_batch_idx + num_return_sequences
                ]
            ):
                generated_decoded[f"answer_{answer_idx}"].append(answer)

    return generated_decoded


def compute_metrics(eval_preds, tokenizer, redirects_on=True):

    preds, labels = eval_preds
    preds = preds.cpu().detach()
    label = labels.cpu().detach()

    # decoding the predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    if redirects_on is True:
        wiki_redirects = WikidataRedirectsCache()

        redirect_labels = []
        for label in decoded_labels:

            # getting the redirect for the current label
            redirects = wiki_redirects.get_redirects(label)
            redirect_labels.append[redirects]

        # calculating bleu score (list('str') vs list(list('str')))
        metric = evaluate.load("sacrebleu")
        result = metric.compute(predictions=decoded_preds, references=redirect_labels)
        return {"bleu": result["score"]}
    else:  # redirects_on is False
        hit1 = 0
        for idx, pred in enumerate(decoded_preds):
            if pred in decoded_labels[idx]:
                hit1 += 1
        return {"Hit@1": hit1 / len(decoded_preds)}


def make_report(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.arrow_dataset.Dataset,
    batch_size: int,
    num_beams: int = 500,
    num_return_sequences: int = 500,
    num_beam_groups: int = 50,
    diversity_penalty: int = 0.1,
    device: torch.device = torch.device("cuda"),
    recall_redirects_on: bool = False,
    question_feature_name: str = "question",
    label_feature_name: str = "object",
) -> Tuple[pd.DataFrame, dict]:
    """make_report

    Args:
        model (PreTrainedModel): model for predicting
        tokenizer (PreTrainedTokenizer): tokenizer for decoding predicted results
        dataset (datasets.arrow_dataset.Dataset): HF dataset
        batch_size (int): batch size for evaluation
        num_return_sequences(int): Number generated answers from beam search process
        question_feature_name (str): Name of column with quesions
        label_feature_name (str): Name of column with labels

    Returns:
        Tuple[pd.DataFrame, dict]: results of predicting with questions on DataFrame and report dict with metrics
    """

    results_df = pd.DataFrame(
        {
            "question": dataset[question_feature_name],
            "target": dataset[label_feature_name],
        }
    )
    generated_answers = predict_answers(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        batch_size=batch_size,
        device=device,
    )
    logger.info("Eval: candidate answers generated")
    results_df = pd.concat([results_df, pd.DataFrame(generated_answers)], axis=1)

    logger.info("Eval: target out of vocab calculation")
    vocab = tokenizer.get_vocab()
    results_df["target_out_of_vocab"] = results_df["target"].apply(
        lambda lbl: vocab["<unk>"] in tokenizer.encode(lbl)
    )
    report = {}
    report["num_out_of_vocab"] = results_df[
        results_df["target_out_of_vocab"]
    ].index.size

    if recall_redirects_on is True:
        logger.info("Eval: recall_redirects_on True. WikidataRedirectsCache used")
        wiki_redirects = WikidataRedirectsCache()
    else:
        logger.info("Eval: recall_redirects_on False.")
        wiki_redirects = None

    for top_k in logging.tqdm(
        range(1, len(generated_answers.keys())), desc="Eval: topN recall calculation"
    ):
        top_k_answers_colums = [f"answer_{k}" for k in range(top_k)]
        report[f"recall_top{top_k}"] = recall(
            results_df["target"].values.tolist(),
            results_df[top_k_answers_colums].values.tolist(),
            wiki_redirects,
        )

    return results_df, report
