import random
import json

import numpy as np
from transformers import set_seed
import torch
import pandas as pd
from collections import Counter

from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm

from ranking_model import (
    NORanker,
    FullRandomRanker,
    LogisticRegressionRanker,
    LinearRegressionRanker,
    RankedAnswer,
    RankedAnswersDict,
)
from ranking_data_utils import prepare_data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse

hf_cache_dir = "/workspace/storage/misc/huggingface"

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
    help="Force reranking even if result files exist."
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
    kgqa_ds_path, data_dir=features_data_dir, cache_dir=hf_cache_dir
)
outputs_ds = load_dataset(
    kgqa_ds_path, data_dir=outputs_data_dir, cache_dir=hf_cache_dir
)
base_ds = load_dataset(base_dataset_path)

train_df = prepare_data(base_ds["train"], outputs_ds["train"], features_ds["train"])
# valid_df = prepare_data(
#     base_ds["validation"], outputs_ds["validation"], features_ds["validation"]
# )
valid_df = None
test_df = prepare_data(base_ds["test"], outputs_ds["test"], features_ds["test"])
test_df.groupby(["id", "question"]).count().describe()

results_path = Path(
    f"./reranking_model_results/{dataset_name}/{ds_type}/"
)
results_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Section 1: NO Ranking (NORanker)
result_file = results_path / f"NO_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"NO_reranking_seq2seq skipped (file exists)")
else:
    no_ranker = NORanker()
    with open(result_file, "w") as f:
        for result in no_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"NO_reranking_seq2seq completed")

# Section 2: FullRandom Ranking
result_file = results_path / f"full_random_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"full_random_reranking_seq2seq skipped (file exists)")
else:
    full_random_ranker = FullRandomRanker()
    with open(result_file, "w") as f:
        for result in full_random_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"full_random_reranking_seq2seq completed")

# Section 3: Semantic Ranking
result_file = results_path / f"semantic_mpnet_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"semantic_mpnet_reranking_seq2seq skipped (file exists)")
else:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    results = []
    groups = test_df.groupby("id")
    
    for question_id, group in tqdm(groups, total=len(test_df["id"].unique())):
        if isinstance(group["graph"].iloc[0], (dict, str)):
            question = group["question"].iloc[0]
            answers = group['question_answer'].apply(lambda x: x.split(";")[-1].strip()).values.tolist()
            answer_entities = group['answerEntity'].values.tolist()
            
            question_embedding = model.encode([question], convert_to_numpy=True)
            answer_embeddings = model.encode(answers, convert_to_numpy=True)
            
            similarities = cosine_similarity(question_embedding, answer_embeddings)[0]
            
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_similarities = [float(s) for s in similarities[sorted_indices]]
            sorted_answer = [answers[i] for i in sorted_indices]
            sorted_answer_entities = [answer_entities[i] for i in sorted_indices]
            
            ranked_answers = [
                RankedAnswer(
                    AnswerEntityID=str(answer_entity_id) if answer_entity_id is not None else None,
                    AnswerString=str(answer),
                    Score=score
                )
                for answer, answer_entity_id, score in zip(sorted_answer, sorted_answer_entities, sorted_similarities)
            ]
        else:
            answers = list(dict.fromkeys(group["model_answers"].iloc[0]).keys())
            ranked_answers = [
                RankedAnswer(
                    AnswerEntityID=None,
                    AnswerString=answer,
                    Score=None
                )
                for answer in answers
            ]
        
        results.append(
            RankedAnswersDict(
                QuestionID=str(question_id),
                RankedAnswers=ranked_answers
            )
        )
    
    with open(result_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"semantic_mpnet_reranking_seq2seq completed")

# Section 4: Majority Vote
result_file = results_path / f"majority_vote_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"majority_vote_reranking_seq2seq skipped (file exists)")
else:
    outputs_test_df = outputs_ds['test'].to_pandas()
    answer_columns = [c for c in outputs_test_df.columns if c.startswith("answer_")]
    
    results = []
    for question_id in tqdm(test_df['id'].unique(), desc="Majority vote"):
        question_row = test_df[test_df['id'] == question_id].iloc[0]
        question = question_row['question']
        
        question_outputs = outputs_test_df[outputs_test_df['question'] == question]
        
        if len(question_outputs) == 0:
            continue
        
        all_answers = []
        for _, row in question_outputs.iterrows():
            for col in answer_columns:
                if col in row and pd.notna(row[col]):
                    answer = str(row[col]).strip()
                    if answer:
                        all_answers.append(answer)
        
        if not all_answers:
            continue
        
        answer_counts = Counter(all_answers)
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        
        ranked_answers = [
            RankedAnswer(
                AnswerEntityID=None,
                AnswerString=answer,
                Score=float(count)
            )
            for answer, count in sorted_answers
        ]
        
        results.append(
            RankedAnswersDict(
                QuestionID=str(question_id),
                RankedAnswers=ranked_answers
            )
        )
    
    with open(result_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"majority_vote_reranking_seq2seq completed. Processed {len(results)} questions.")

# Section 5: Logistic Regression
# 1. text only
result_file = results_path / f"logreg_text_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"logreg_text_reranking_seq2seq skipped (file exists)")
else:
    logreg_ranker = LogisticRegressionRanker(sequence_features=features_map["text"])
    logreg_ranker.fit(train_df, n_jobs=16)
    with open(result_file, "w") as f:
        for result in logreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"logreg_text_reranking_seq2seq completed")

# 2. graph only (with scaler)
result_file = results_path / f"logreg_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"logreg_graph_reranking_seq2seq skipped (file exists)")
else:
    logreg_ranker = LogisticRegressionRanker(graph_features=features_map["graph"])
    logreg_ranker.fit(train_df, n_jobs=16, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in logreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"logreg_graph_reranking_seq2seq completed")

# 3. text + graph (with scaler)
result_file = results_path / f"logreg_text_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"logreg_text_graph_reranking_seq2seq skipped (file exists)")
else:
    logreg_ranker = LogisticRegressionRanker(
        sequence_features=features_map["text"], graph_features=features_map["graph"]
    )
    logreg_ranker.fit(train_df, n_jobs=16, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in logreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"logreg_text_graph_reranking_seq2seq completed")

# 4. g2t_determ only
result_file = results_path / f"logreg_g2t_determ_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"logreg_g2t_determ_reranking_seq2seq skipped (file exists)")
else:
    logreg_ranker = LogisticRegressionRanker(sequence_features=features_map["g2t_determ"])
    logreg_ranker.fit(train_df, n_jobs=16)
    with open(result_file, "w") as f:
        for result in logreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"logreg_g2t_determ_reranking_seq2seq completed")

# 5. g2t_t5 only (mintaka only)
if dataset_name == "mintaka":
    result_file = results_path / f"logreg_g2t_t5_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"logreg_g2t_t5_reranking_seq2seq skipped (file exists)")
    else:
        logreg_ranker = LogisticRegressionRanker(sequence_features=features_map["g2t_t5"])
        logreg_ranker.fit(train_df, n_jobs=16)
        with open(result_file, "w") as f:
            for result in logreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"logreg_g2t_t5_reranking_seq2seq completed")

# 6. g2t_gap only (mintaka only)
if dataset_name == "mintaka":
    result_file = results_path / f"logreg_g2t_gap_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"logreg_g2t_gap_reranking_seq2seq skipped (file exists)")
    else:
        logreg_ranker = LogisticRegressionRanker(sequence_features=features_map["g2t_gap"])
        logreg_ranker.fit(train_df, n_jobs=16)
        with open(result_file, "w") as f:
            for result in logreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"logreg_g2t_gap_reranking_seq2seq completed")

# 7. text + g2t_determ + graph (with scaler)
result_file = results_path / f"logreg_text_g2t_determ_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"logreg_text_g2t_determ_graph_reranking_seq2seq skipped (file exists)")
else:
    logreg_ranker = LogisticRegressionRanker(
        sequence_features=features_map["text"] + features_map["g2t_determ"],
        graph_features=features_map["graph"],
    )
    logreg_ranker.fit(train_df, n_jobs=16, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in logreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"logreg_text_g2t_determ_graph_reranking_seq2seq completed")

# 8. text + g2t_t5 + graph (mintaka only, with scaler)
if dataset_name == "mintaka":
    result_file = results_path / f"logreg_text_g2t_t5_graph_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"logreg_text_g2t_t5_graph_reranking_seq2seq skipped (file exists)")
    else:
        logreg_ranker = LogisticRegressionRanker(
            sequence_features=features_map["text"] + features_map["g2t_t5"],
            graph_features=features_map["graph"],
        )
        logreg_ranker.fit(train_df, n_jobs=16, scaler=preprocessing.MinMaxScaler())
        with open(result_file, "w") as f:
            for result in logreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"logreg_text_g2t_t5_graph_reranking_seq2seq completed")

# 9. text + g2t_gap + graph (mintaka only, with scaler)
if dataset_name == "mintaka":
    result_file = results_path / f"logreg_text_g2t_gap_graph_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"logreg_text_g2t_gap_graph_reranking_seq2seq skipped (file exists)")
    else:
        logreg_ranker = LogisticRegressionRanker(
            sequence_features=features_map["text"] + features_map["g2t_gap"],
            graph_features=features_map["graph"],
        )
        logreg_ranker.fit(train_df, n_jobs=16, scaler=preprocessing.MinMaxScaler())
        with open(result_file, "w") as f:
            for result in logreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"logreg_text_g2t_gap_graph_reranking_seq2seq completed")

# Section 6: Linear Regression
# 1. text only
result_file = results_path / f"linreg_text_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"linreg_text_reranking_seq2seq skipped (file exists)")
else:
    linreg_ranker = LinearRegressionRanker(sequence_features=features_map["text"])
    linreg_ranker.fit(train_df, n_jobs=8)
    with open(result_file, "w") as f:
        for result in linreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"linreg_text_reranking_seq2seq completed")

# 2. graph only (with scaler)
result_file = results_path / f"linreg_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"linreg_graph_reranking_seq2seq skipped (file exists)")
else:
    linreg_ranker = LinearRegressionRanker(graph_features=features_map["graph"])
    linreg_ranker.fit(train_df, n_jobs=8, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in linreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"linreg_graph_reranking_seq2seq completed")

# 3. text + graph (with scaler)
result_file = results_path / f"linreg_text_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"linreg_text_graph_reranking_seq2seq skipped (file exists)")
else:
    linreg_ranker = LinearRegressionRanker(
        sequence_features=features_map["text"], graph_features=features_map["graph"]
    )
    linreg_ranker.fit(train_df, n_jobs=8, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in linreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"linreg_text_graph_reranking_seq2seq completed")

# 4. g2t_determ only
result_file = results_path / f"linreg_g2t_determ_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"linreg_g2t_determ_reranking_seq2seq skipped (file exists)")
else:
    linreg_ranker = LinearRegressionRanker(sequence_features=features_map["g2t_determ"])
    linreg_ranker.fit(train_df, n_jobs=8)
    with open(result_file, "w") as f:
        for result in linreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"linreg_g2t_determ_reranking_seq2seq completed")

# 5. g2t_t5 only (mintaka only)
if dataset_name == "mintaka":
    result_file = results_path / f"linreg_g2t_t5_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"linreg_g2t_t5_reranking_seq2seq skipped (file exists)")
    else:
        linreg_ranker = LinearRegressionRanker(sequence_features=features_map["g2t_t5"])
        linreg_ranker.fit(train_df, n_jobs=8)
        with open(result_file, "w") as f:
            for result in linreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"linreg_g2t_t5_reranking_seq2seq completed")

# 6. g2t_gap only (mintaka only)
if dataset_name == "mintaka":
    result_file = results_path / f"linreg_g2t_gap_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"linreg_g2t_gap_reranking_seq2seq skipped (file exists)")
    else:
        linreg_ranker = LinearRegressionRanker(sequence_features=features_map["g2t_gap"])
        linreg_ranker.fit(train_df, n_jobs=8)
        with open(result_file, "w") as f:
            for result in linreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"linreg_g2t_gap_reranking_seq2seq completed")

# 7. text + g2t_determ + graph (with scaler)
result_file = results_path / f"linreg_text_g2t_determ_graph_reranking_seq2seq_{ds_type}_results.jsonl"
if not force and result_file.exists():
    print(f"linreg_text_g2t_determ_graph_reranking_seq2seq skipped (file exists)")
else:
    linreg_ranker = LinearRegressionRanker(
        sequence_features=features_map["text"] + features_map["g2t_determ"],
        graph_features=features_map["graph"],
    )
    linreg_ranker.fit(train_df, n_jobs=8, scaler=preprocessing.MinMaxScaler())
    with open(result_file, "w") as f:
        for result in linreg_ranker.rerank(test_df):
            f.write(json.dumps(result) + "\n")
    print(f"linreg_text_g2t_determ_graph_reranking_seq2seq completed")

# 8. text + g2t_t5 + graph (mintaka only, with scaler)
if dataset_name == "mintaka":
    result_file = results_path / f"linreg_text_g2t_t5_graph_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"linreg_text_g2t_t5_graph_reranking_seq2seq skipped (file exists)")
    else:
        linreg_ranker = LinearRegressionRanker(
            sequence_features=features_map["text"] + features_map["g2t_t5"],
            graph_features=features_map["graph"],
        )
        linreg_ranker.fit(train_df, n_jobs=8, scaler=preprocessing.MinMaxScaler())
        with open(result_file, "w") as f:
            for result in linreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"linreg_text_g2t_t5_graph_reranking_seq2seq completed")

# 9. text + g2t_gap + graph (mintaka only, with scaler)
if dataset_name == "mintaka":
    result_file = results_path / f"linreg_text_g2t_gap_graph_reranking_seq2seq_{ds_type}_results.jsonl"
    if not force and result_file.exists():
        print(f"linreg_text_g2t_gap_graph_reranking_seq2seq skipped (file exists)")
    else:
        linreg_ranker = LinearRegressionRanker(
            sequence_features=features_map["text"] + features_map["g2t_gap"],
            graph_features=features_map["graph"],
        )
        linreg_ranker.fit(train_df, n_jobs=8, scaler=preprocessing.MinMaxScaler())
        with open(result_file, "w") as f:
            for result in linreg_ranker.rerank(test_df):
                f.write(json.dumps(result) + "\n")
        print(f"linreg_text_g2t_gap_graph_reranking_seq2seq completed")
