"""label_subgraphs_dataset - script for adding labels to subgraph dataset
"""
import argparse
from copy import deepcopy

import ujson
from joblib import Parallel, delayed
from pywikidata import Entity
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_jsonl_path",
    default="/workspace/storage/new_subgraph_dataset/t5-xl-ssm/mintaka_test.jsonl",
    type=str,
    help="Path to dataset in JSONL format without labels",
)

parser.add_argument(
    "--save_jsonl_path",
    default="/workspace/storage/new_subgraph_dataset/t5-xl-ssm/mintaka_test_labeled.jsonl",
    type=str,
    help="Path to resulting JSONL: subgraphs_dataset_prepared_entities_jsonl_path and graph",
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=8,
    help="Number of parallel process",
)


def label_record(record):
    """label_record - method for adding labels for subgraph dataset objects"""
    record = deepcopy(record)
    for node in record["graph"]["nodes"]:
        node["label"] = Entity(node["name_"]).label
    for link in record["graph"]["links"]:
        link["label"] = Entity(link["name_"]).label
    return record


if __name__ == "__main__":
    args = parser.parse_args()

    data = []
    with open(args.dataset_jsonl_path, "r", encoding="UTF-8") as f:
        for line in f:
            data.append(ujson.loads(line))

    labeled_data = Parallel(n_jobs=args.n_jobs)(
        delayed(label_record)(record)
        for record in tqdm(data, desc="Labeling subgraph dataset")
    )

    with open(args.save_jsonl_path, "w", encoding="UTF-8") as f:
        for line in labeled_data:
            f.write(ujson.dumps(line) + "\n")
