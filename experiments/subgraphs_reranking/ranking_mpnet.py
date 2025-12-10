import random
import json

import numpy as np
from transformers import set_seed
import torch

from pathlib import Path
from datasets import load_dataset

from ranking_model import MPNetRanker
from ranking_data_utils import prepare_data

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# os.environ['HF_DATASETS_CACHE'] = '/workspace/storage/misc/huggingface'

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
base_path = "/mnt/storage/QA_System_Project/kbqa_reranking_experiments_runs/sequence/"

seq_type_configs = {
    "question_answer": "text_only_determ",
    "no_highlighted_determ_sequence": "no_hl_g2t_determ",
    "highlighted_determ_sequence": "hl_g2t_determ",
    "no_highlighted_t5_sequence": "no_hl_g2t_t5",
    "highlighted_t5_sequence": "hl_g2t_t5",
    "no_highlighted_gap_sequence": "no_hl_g2t_gap",
    "highlighted_gap_sequence": "hl_g2t_gap",
}

if dataset_name == "mintaka":
    sequence_types = [
        "question_answer",
        "no_highlighted_determ_sequence",
        "highlighted_determ_sequence",
        "no_highlighted_t5_sequence",
        "highlighted_t5_sequence",
        "no_highlighted_gap_sequence",
        "highlighted_gap_sequence",
    ]
elif dataset_name == "mkqa-hf":
    sequence_types = [
        "question_answer",
        "no_highlighted_determ_sequence",
    ]

for run in [1, 2, 3]:
    random.seed(42 + run)
    np.random.seed(42 + run)
    torch.manual_seed(42 + run)
    torch.cuda.manual_seed_all(42 + run)
    set_seed(42 + run)
    
    for seq_type in sequence_types:
        result_suffix = seq_type_configs[seq_type]
        result_file = results_path / f"mpnet_{result_suffix}_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl"
        
        if not force and result_file.exists():
            print(f"RUN {run} mpnet_{result_suffix}_reranking_seq2seq skipped (file exists)")
            continue
        
        model_path = Path(base_path) / dataset_name / seq_type / ds_type / f"run_{run}" / "outputs" / "checkpoint-best"
        
        if not model_path.exists():
            print(f"RUN {run} mpnet_{result_suffix}_reranking_seq2seq skipped (model not found: {model_path})")
            continue
        
        try:
            mpnet_ranker = MPNetRanker(seq_type, str(model_path), device)
            with open(result_file, "w") as f:
                for result in mpnet_ranker.rerank(test_df):
                    f.write(json.dumps(result) + "\n")
            print(f"RUN {run} mpnet_{result_suffix}_reranking_seq2seq completed")
        except Exception as e:
            print(f"RUN {run} mpnet_{result_suffix}_reranking_seq2seq failed: {e}")
            continue
