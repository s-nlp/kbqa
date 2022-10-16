# pylint: disable=fixme,unused-import

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie
from tqdm.auto import tqdm

from caches.genre import GENREWikidataEntityesCache
from caches.ner_to_sentence_insertion import NerToSentenceInsertion
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity
from wikidata.wikidata_redirects import WikidataRedirectsCache
from wikidata.wikidata_shortest_path import WikidataShortestPathCache
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever

parser = argparse.ArgumentParser()
parser.add_argument(
    "--candidates_results_path",
    default="./subgraphs_dataset/results.csv",
    type=str,
)
parser.add_argument(
    "--lang_title_wikidata_id_path",
    default="./lang_title2wikidataID-normalized_with_redirect.pkl",
    type=str,
)
parser.add_argument(
    "--marisa_trie_path",
    default="./titles_lang_all105_marisa_trie_with_redirect.pkl",
    type=str,
)
parser.add_argument(
    "--ner_model_path",
    default="../ner_model/",
    type=str,
)
parser.add_argument(
    "--pretrained_mgenre_weight_path",
    default="./fairseq_multilingual_entity_disambiguation",
    type=str,
)
parser.add_argument(
    "--edge_between_path",
    default=True,
    type=bool,
)
parser.add_argument("--save_dir", default="./subgraphs_dataset/dataset_v0/")
parser.add_argument(
    "--number_of_pathes",
    default=3,
    type=int,
    help="maximum number of shortest pathes that will queried from KG for entities pair",
)
parser.add_argument(
    "--genre_batch_size",
    default=10,
    type=int,
)


def load_pkl(lang_title_wikidata_id_path, marisa_trie_path):
    """
    load the needed pkl files
    """
    with open(lang_title_wikidata_id_path, "rb") as file_handler:
        lang_title_wikidata_id = pickle.load(file_handler)

    with open(marisa_trie_path, "rb") as file_handler:
        trie = pickle.load(file_handler)

    print("finished loading pkl")
    return lang_title_wikidata_id, trie


def load_model(path):
    """
    load our mGENRE
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mGENRE.from_pretrained(path).eval()
    model.to(device)
    print("mGENRE loaded")
    return model


def chunks(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def get_entites_batch(genre_entities, questions_arr, batch_size):
    """
    get all entites batches
    """
    # divide our questions into n size batches and get the entites
    questions_arr_batches = list(chunks(questions_arr, batch_size))
    generated_entities_batches = []

    # get entity for each batch
    for batch in tqdm(questions_arr_batches, 'GENRE'):
        generated_entities_batch = genre_entities.sentences_batch_to_entities(batch)
        generated_entities_batches.extend(generated_entities_batch)

    return generated_entities_batches


def prepare_data(results_df, label2entity, ner, genre_entities, genre_batch_size):
    results_df["question_with_ner"] = results_df["question"].apply(ner.entity_labeling)
    results_df["question_entities"] = get_entites_batch(
        genre_entities, results_df["question_with_ner"].values.tolist(), genre_batch_size
    )

    answers_cols = [col for col in results_df.columns if "answer_" in col]

    data_collection = defaultdict(list)

    for _, row in tqdm(
        results_df.iterrows(), total=results_df.index.size, desc="Prepare data"
    ):
        target_id = label2entity.get_id(row["target"])

        question = row["question_with_ner"]
        question_entities = row["question_entities"]

        candidates = np.unique(row[["target"] + answers_cols].values).tolist()
        for candidate in candidates:
            candidate_id = label2entity.get_id(candidate)
            if candidate_id and candidate_id != "":
                data_collection["question"].append(question)
                data_collection["target"].append(target_id)
                data_collection["candidate"].append(candidate_id)
                data_collection["is_correct"].append(target_id == candidate_id)

                # TODO: Add Th filter for entities score
                data_collection["question_entities"].append(
                    [qe["id"] for qe in question_entities]
                )

                data_collection["raw_question"].append(row["question"])
                data_collection["raw_target"].append(row["target"])
                data_collection["raw_candidate"].append(candidate)

    return pd.DataFrame(data_collection)


if __name__ == "__main__":
    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=False)
    with open(Path(args.save_dir) / "dataset_info.json", "w") as file_handler:
        json.dump(vars(args), file_handler)

    lang_title_wikidata_id, trie = load_pkl(
        args.lang_title_wikidata_id_path, args.marisa_trie_path
    )
    genre_model = load_model(args.pretrained_mgenre_weight_path)
    genre_entities = GENREWikidataEntityesCache(
        genre_model,
        trie,
        lang_title_wikidata_id,
    )

    wd_redirects = WikidataRedirectsCache()
    label2entity = WikidataLabelToEntity(wd_redirects)
    ner = NerToSentenceInsertion(model_path=args.ner_model_path)
    shortest_path = WikidataShortestPathCache()
    entity2label = WikidataEntityToLabel()
    subgraph_retriever = SubgraphsRetriever(
        entity2label=entity2label,
        shortest_path=shortest_path,
        edge_between_path=args.edge_between_path,
    )

    results_df = pd.read_csv(args.candidates_results_path)
    data_collection_df = prepare_data(
        results_df, label2entity, ner, genre_entities, args.genre_batch_size
    )
    print("Count of graphs", data_collection_df.index.size)
    print(
        "Correct answer samples",
        data_collection_df[data_collection_df["is_correct"]].index.size,
    )

    for idx, row in tqdm(
        data_collection_df.iterrows(),
        total=data_collection_df.index.size,
        desc="subgraphs_dataset_generation",
    ):
        graph = subgraph_retriever.get_subgraph(
            row["question_entities"], row["candidate"], args.number_of_pathes
        )

        graph_save_path = Path(args.save_dir) / f"{idx}"
        graph_save_path.mkdir(parents=True, exist_ok=True)
        with open(graph_save_path / "graph.json", "w") as file_handler:
            json.dump(nx.node_link_data(graph), file_handler)
        with open(graph_save_path / "meta.json", "w") as file_handler:
            json.dump(
                {
                    "_idx": idx,
                    "question": row["question"],
                    "candidate": row["candidate"],
                    "target": row["target"],
                    "raw_question": row["raw_question"],
                    "raw_target": row["raw_target"],
                    "raw_candidate": row["raw_candidate"],
                },
                file_handler,
            )
