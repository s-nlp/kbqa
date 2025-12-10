import os
import pandas as pd
from datasets import load_dataset, Dataset
import random
import json

import numpy as np
from transformers import set_seed
import torch
from tqdm.auto import tqdm

from pathlib import Path

from ranking_model import (
    LogisticRegressionRanker,
    LinearRegressionRanker,
    MPNetRanker,
    CatboostRanker,
    FullRandomRanker,
    NORanker,
)
from ranking_data_utils import prepare_data
from sklearn import preprocessing

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# os.environ['HF_DATASETS_CACHE'] = '/workspace/storage/misc/huggingface'

import argparse

parser = argparse.ArgumentParser(description="Subgraphs Reranking")
parser.add_argument(
    "--ds_type",
    type=str,
    default="t5xlssm",
    choices=["t5largessm", "t5xlssm", "mistral", "mixtral"],
    help="Type of dataset/features to use."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mintaka",
    choices=["mintaka", "mkqa-hf"],
    help="Dataset to use: mintaka or mkqa-hf."
)
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="Force retraining even if result files exist."
)
args = parser.parse_args()

ds_type = args.ds_type  # 't5largessm', 't5xlssm', 'mistral' or 'mixtral'
dataset_name = args.dataset  # 'mintaka' or 'mkqa-hf'
force = args.force

if dataset_name == "mintaka":
    base_dataset_path = "AmazonScience/mintaka"
    kgqa_ds_path = "s-nlp/KGQASubgraphsRanking"
    features_data_dir = f"{ds_type}_subgraphs"
    outputs_data_dir = f"{ds_type}_outputs"
elif dataset_name == "mkqa-hf":
    base_dataset_path = "Dms12/mkqa_mintaka_format_with_question_entities"
    kgqa_ds_path = "s-nlp/MKQASubgraphsRanking"
    features_data_dir = f"mkqa_{ds_type}_subgraphs"
    outputs_data_dir = f"mkqa_{ds_type}_outputs"

features_ds = load_dataset(
    kgqa_ds_path, data_dir=features_data_dir
)
outputs_ds = load_dataset(
    kgqa_ds_path, data_dir=outputs_data_dir
)
base_ds = load_dataset(base_dataset_path)

train_df = prepare_data(base_ds["train"], outputs_ds["train"], features_ds["train"])
# valid_df = prepare_data(
#     base_ds["validation"], outputs_ds["validation"], features_ds["validation"]
# )
valid_df = None
test_df = prepare_data(base_ds["test"], outputs_ds["test"], features_ds["test"])
test_df.groupby(["id", "question"]).count().describe()

features_map = {
    "text": ["question_answer_embedding"],
    "graph": [
        "num_nodes",
        "num_edges",
        "density",
        "cycle",
        "bridge",
        "katz_centrality",
        "page_rank",
        "avg_ssp_length",
    ],
}
if dataset_name == "mintaka":
    features_map["g2t_determ"] = ["determ_sequence_embedding"]
    features_map["g2t_t5"] = ["t5_sequence_embedding"]
    features_map["g2t_gap"] = ["gap_sequence_embedding"]
elif dataset_name == "mkqa-hf":
    features_map["g2t_determ"] = ["no_highlighted_determ_sequence_embedding"]
    # features_map["g2t_t5"] = ["t5_sequence_embedding"]
    # features_map["g2t_gap"] = ["gap_sequence_embedding"]

results_path = Path(
    f"./reranking_model_results/{dataset_name}/{ds_type}/"
)
results_path.mkdir(parents=True, exist_ok=True)


catboost_dir = (
    f"/mnt/storage/QA_System_Project/kbqa_reranking_experiments_runs/catboost/{dataset_name}/{ds_type}"
)

for run in [1, 2, 3]:
    random.seed(42 + run)
    np.random.seed(42 + run)
    torch.manual_seed(42 + run)
    torch.cuda.manual_seed_all(42 + run)
    set_seed(42 + run)
    
    result_file = results_path / f"catboost_text_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"RUN {run} catboost_text_reranking_seq2seq skipped (file exists)")
    else:
        Path(f"{catboost_dir}/text/run_{run}").mkdir(parents=True, exist_ok=True)
        model_weights = f"{catboost_dir}/text/run_{run}/best_model"
        catboost_ranker = CatboostRanker(sequence_features=features_map["text"])
        catboost_ranker.fit(train_df, val_df=valid_df, model_save_path=model_weights)
        with open(result_file, "w") as f:
            for result in catboost_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"RUN {run} catboost_text_reranking_seq2seq completed")

    result_file = results_path / f"catboost_graph_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"RUN {run} catboost_graph_reranking_seq2seq skipped (file exists)")
    else:
        Path(f"{catboost_dir}/graph/run_{run}").mkdir(parents=True, exist_ok=True)
        model_weights = f"{catboost_dir}/graph/run_{run}/best_model"
        fitted_scaler_path = f"{catboost_dir}/graph/run_{run}/fitted_scaler.bz2"
        catboost_ranker = CatboostRanker(graph_features=features_map["graph"])
        catboost_ranker.fit(
            train_df,
            val_df=valid_df,
            model_save_path=model_weights,
            scaler_save_path=fitted_scaler_path,
        )
        with open(result_file, "w") as f:
            for result in catboost_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"RUN {run} catboost_graph_reranking_seq2seq completed")

    result_file = results_path / f"catboost_text_graph_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"RUN {run} catboost_text_graph_reranking_seq2seq skipped (file exists)")
    else:
        Path(f"{catboost_dir}/text_graph/run_{run}").mkdir(parents=True, exist_ok=True)
        model_weights = f"{catboost_dir}/text_graph/run_{run}/best_model"
        fitted_scaler_path = f"{catboost_dir}/text_graph/run_{run}/fitted_scaler.bz2"
        catboost_ranker = CatboostRanker(
            sequence_features=features_map["text"],
            graph_features=features_map["graph"],
        )
        catboost_ranker.fit(
            train_df,
            val_df=valid_df,
            model_save_path=model_weights,
            scaler_save_path=fitted_scaler_path,
        )
        with open(result_file, "w") as f:
            for result in catboost_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"RUN {run} catboost_text_graph_reranking_seq2seq completed")

    result_file = results_path / f"catboost_g2t_determ_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"RUN {run} catboost_g2t_determ_reranking_seq2seq skipped (file exists)")
    else:
        Path(f"{catboost_dir}/g2t_determ/run_{run}").mkdir(parents=True, exist_ok=True)
        model_weights = f"{catboost_dir}/g2t_determ/run_{run}/best_model"
        catboost_ranker = CatboostRanker(sequence_features=features_map["g2t_determ"])
        catboost_ranker.fit(train_df, val_df=valid_df, model_save_path=model_weights)
        with open(result_file, "w") as f:
            for result in catboost_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"RUN {run} catboost_g2t_determ_reranking_seq2seq completed")

    if dataset_name == "mintaka":
        result_file = results_path / f"catboost_g2t_t5_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
        if not force and result_file.exists():
            print(f"RUN {run} catboost_g2t_t5_reranking_seq2seq skipped (file exists)")
        else:
            Path(f"{catboost_dir}/g2t_t5/run_{run}").mkdir(parents=True, exist_ok=True)
            model_weights = f"{catboost_dir}/g2t_t5/run_{run}/best_model"
            catboost_ranker = CatboostRanker(sequence_features=features_map["g2t_t5"])
            catboost_ranker.fit(train_df, val_df=valid_df, model_save_path=model_weights)
            with open(result_file, "w") as f:
                for result in catboost_ranker.rerank(test_df):
                    f.write(json.dumps(result) + "\n")
            print(f"RUN {run} catboost_g2t_t5_reranking_seq2seq completed")

        result_file = results_path / f"catboost_g2t_gap_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
        if not force and result_file.exists():
            print(f"RUN {run} catboost_g2t_gap_reranking_seq2seq skipped (file exists)")
        else:
            Path(f"{catboost_dir}/g2t_gap/run_{run}").mkdir(parents=True, exist_ok=True)
            model_weights = f"{catboost_dir}/g2t_gap/run_{run}/best_model"
            catboost_ranker = CatboostRanker(sequence_features=features_map["g2t_gap"])
            catboost_ranker.fit(train_df, val_df=valid_df, model_save_path=model_weights)
            with open(result_file, "w") as f:
                for result in catboost_ranker.rerank(test_df):
                    f.write(json.dumps(result) + "\n")
            print(f"RUN {run} catboost_g2t_gap_reranking_seq2seq completed")

    result_file = results_path / f"catboost_text_graph_g2t_determ_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"RUN {run} catboost_text_graph_g2t_determ_reranking_seq2seq skipped (file exists)")
    else:
        Path(f"{catboost_dir}/text_graph_g2t_determ/run_{run}").mkdir(parents=True, exist_ok=True)
        model_weights = f"{catboost_dir}/text_graph_g2t_determ/run_{run}/best_model"
        fitted_scaler_path = f"{catboost_dir}/text_graph_g2t_determ/run_{run}/fitted_scaler.bz2"
        catboost_ranker = CatboostRanker(
            sequence_features=features_map["text"] + features_map["g2t_determ"],
            graph_features=features_map["graph"],
        )
        catboost_ranker.fit(
            train_df,
            val_df=valid_df,
            model_save_path=model_weights,
            scaler_save_path=fitted_scaler_path,
        )
        with open(result_file, "w") as f:
            for result in catboost_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"RUN {run} catboost_text_graph_g2t_determ_reranking_seq2seq completed")

    if dataset_name == "mintaka":
        result_file = results_path / f"catboost_text_graph_g2t_t5_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
        if not force and result_file.exists():
            print(f"RUN {run} catboost_text_graph_g2t_t5_reranking_seq2seq skipped (file exists)")
        else:
            Path(f"{catboost_dir}/text_graph_g2t_t5/run_{run}").mkdir(parents=True, exist_ok=True)
            model_weights = f"{catboost_dir}/text_graph_g2t_t5/run_{run}/best_model"
            fitted_scaler_path = f"{catboost_dir}/text_graph_g2t_t5/run_{run}/fitted_scaler.bz2"
            catboost_ranker = CatboostRanker(
                sequence_features=features_map["text"] + features_map["g2t_t5"],
                graph_features=features_map["graph"],
            )
            catboost_ranker.fit(
                train_df,
                val_df=valid_df,
                model_save_path=model_weights,
                scaler_save_path=fitted_scaler_path,
            )
            with open(result_file, "w") as f:
                for result in catboost_ranker.rerank(test_df):
                    f.write(json.dumps(result) + "\n")
            print(f"RUN {run} catboost_text_graph_g2t_t5_reranking_seq2seq completed")

        result_file = results_path / f"catboost_text_graph_g2t_gap_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
        if not force and result_file.exists():
            print(f"RUN {run} catboost_text_graph_g2t_gap_reranking_seq2seq skipped (file exists)")
        else:
            Path(f"{catboost_dir}/text_graph_g2t_gap/run_{run}").mkdir(parents=True, exist_ok=True)
            model_weights = f"{catboost_dir}/text_graph_g2t_gap/run_{run}/best_model"
            fitted_scaler_path = f"{catboost_dir}/text_graph_g2t_gap/run_{run}/fitted_scaler.bz2"
            catboost_ranker = CatboostRanker(
                sequence_features=features_map["text"] + features_map["g2t_gap"],
                graph_features=features_map["graph"],
            )
            catboost_ranker.fit(
                train_df,
                val_df=valid_df,
                model_save_path=model_weights,
                scaler_save_path=fitted_scaler_path,
            )
            with open(result_file, "w") as f:
                for result in catboost_ranker.rerank(test_df):
                    f.write(json.dumps(result) + "\n")
            print(f"RUN {run} catboost_text_graph_g2t_gap_reranking_seq2seq completed")
