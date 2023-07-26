"""Apply ranking model for rerank subgraphs
"""
import argparse
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import pandas as pd
from data_utils import data_df_convert
from tqdm import tqdm
import numpy as np
from typing import Dict, List
from pywikidata.utils import get_wd_search_results
from evaluateqa.mintaka import calculate_metrics_for_prediction
from evaluateqa.mintaka import evaluate as evaluate_mintaka
import ujson
import json
from pprint import pprint


parse = argparse.ArgumentParser()
parse.add_argument("model_path", type=str, help="Path to pretrained model")
parse.add_argument("subgraph_path", type=str, help="Path to JSONL file with subgraphs")
parse.add_argument(
    "seq2seq_candidates_path",
    type=str,
    help="Path to csv file with seq2seq predicted labels (results.csv)",
)
parse.add_argument(
    "--classification_threshold",
    type=float,
    default=0.5,
    help="Threashold for classification prediction. From 0 to 1, Default 0.5",
)
parse.add_argument(
    "--ranked_preds_save_path",
    default="./ranked_preds.json",
    help="Path to json file with predictions",
)
parse.add_argument(
    "--ranked_preds_metrics_path",
    default="./ranked_preds_metrics.json",
    help="Path to file with metrics",
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def apply_ranker_model(
    subgraph_df, model, tokenizer, device, classification_threshold
) -> Dict[str, List[str]]:
    ranked_preds = {}
    for qid, group in tqdm(
        subgraph_df.groupby("id"), total=len(subgraph_df["id"].unique())
    ):
        batch = tokenizer(
            group["graph_sequence"].values.tolist(),
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            preds = (
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                .logits.view(-1)
                .detach()
                .cpu()
                .numpy()
            )

        ranked_ids = np.argsort(preds)[::-1]
        if preds[ranked_ids[0]] >= classification_threshold:
            ranked_top1_answer = group["answerEntity"].values[ranked_ids[0]]
            ranked_preds[qid] = ranked_top1_answer
    return ranked_preds


def str_answer_to_kg(answer):
    if not isinstance(answer, str):
        return None

    if answer.lower() in ["yes", "true"]:
        return True
    elif answer.lower() in ["no", "false"]:
        return False

    try:
        answer = float(answer)
        if answer == round(answer):
            answer = int(answer)
        return answer
    except:
        pass

    try:
        return get_wd_search_results(answer)[0]
    except:
        pass

    return None


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    subgraph_df = pd.read_json(args.subgraph_path, lines=True)
    subgraph_df = data_df_convert(subgraph_df, sep_token=tokenizer.sep_token)

    ranked_preds = apply_ranker_model(
        subgraph_df,
        model,
        tokenizer,
        device,
        args.classification_threshold,
    )
    metrics_df = calculate_metrics_for_prediction(ranked_preds, split="test", mode="kg")

    # Add seq2seq answer if ranker not predicted top1 answer from subgraph
    df = pd.read_csv(args.seq2seq_candidates_path)
    df = pd.merge(left=metrics_df, right=df, on="question")

    for _, row in tqdm(df.iterrows(), total=df.index.size):
        if row["id"] not in ranked_preds:
            new_answer = str_answer_to_kg(row["answer_0"])
            ranked_preds[row["id"]] = [new_answer]

    with open(args.ranked_preds_save_path, "w") as f:
        ujson.dump(ranked_preds, f)
    print("Ranked predictions dumped to ", args.ranked_preds_save_path)

    results = evaluate_mintaka(
        predictions=ranked_preds,
        mode="kg",
    )
    with open(args.ranked_preds_metrics_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    print("Results")
    pprint(results)


if __name__ == "__main__":
    args = parse.parse_args()
    main(args)
