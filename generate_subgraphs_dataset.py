import pandas as pd
import pickle
import torch
import argparse
import json
from tqdm import tqdm
import shutil
from pathlib import Path
import networkx as nx

from caches.ner_to_sentence_insertion import NerToSentenceInsertion
from genre.fairseq_model import mGENRE  # pylint: disable=import-error,import-error
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from caches.genre import GENREWikidataEntityesCache
from subgraphs_dataset.question_entities_candidate import QuestionEntitiesCandidates
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever
from wikidata.wikidata_shortest_path import WikidataShortestPathCache
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity
from wikidata.wikidata_redirects import WikidataRedirectsCache
from genre.trie import Trie, MarisaTrie  # pylint: disable=unused-import,import-error
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_bad_candidates",
    default=5,
    type=int,
)
parser.add_argument(
    "--model_result_path",
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
    "--pretrained_mgenre_weight_path",
    default="./fairseq_multilingual_entity_disambiguation",
    type=str,
)
parser.add_argument(
    "--ner_model_path",
    default="../ner_model/",
    type=str,
)
parser.add_argument(
    "--edge_between_path",
    default=True,
    type=bool,
)
parser.add_argument(
    "--number_of_pathes",
    default=3,
    type=int,
    help="maximum number of shortest pathes that will queried from KG for entities pair",
)
parser.add_argument(
    "--dataset_extension_type",
    default="json",
    type=str,
    choices=["json", "pkl"],
)
parser.add_argument(
    "--batch_size",
    default=50,
    type=int,
)
parser.add_argument(
    "--num_rows",
    default=500,
    type=int,
)
parser.add_argument("--save_dir", default="./subgraphs_dataset/dataset_v0/", type=str)


def prepare_csv(path, num_rows):
    """
    return the pd df with num_ans answers to our question
    """
    df = pd.read_csv(path)
    print("loaded results.csv")
    return df.head(num_rows)


def load_pkl(lang_title_wikidata_id_path, marisa_trie_path):
    """
    load the needed pkl files
    """
    with open(lang_title_wikidata_id_path, "rb") as f:
        lang_title_wikidata_id = pickle.load(f)

    with open(marisa_trie_path, "rb") as f:
        trie = pickle.load(f)

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


def get_entities_batch(genre_entities, questions_arr, batch_size):
    """
    get all entites batches
    """
    # divide our questions into n size batches and get the entites
    questions_arr_batches = list(chunks(questions_arr, batch_size))
    generated_entities_batches = []

    # get entity for each batch
    for batch in questions_arr_batches:
        generated_entities_batch = genre_entities.sentences_batch_to_entities(batch)
        generated_entities_batches.append(generated_entities_batch)

    return generated_entities_batches


def prepare_questions_entities_candidates(
    df,
    questions_entities_candidates,
    genre_entities,
    label2entity,
    batch_size,
    num_ans,
):
    """
    prepare all questions, question entities and candidates for subgraph generation
    """
    # getting candidates df
    candidates_df = df.drop(columns=df.columns[:1])
    ner = NerToSentenceInsertion(model_path=args.ner_model_path)

    def _get_ner_sentence_entities(df):
        ner_sentence, ner_entities = ner.entity_labeling(
            df["question"], return_entities_list=True
        )
        df["ner_question"] = ner_sentence
        df["ner_entities"] = len(ner_entities)
        return df

    ner_df = df.apply(_get_ner_sentence_entities, axis=1)

    # getting ner questions & num entities
    original_questions = ner_df["question"].values.tolist()
    ner_questions = ner_df["ner_question"].values.tolist()
    ner_num_entities = ner_df["ner_entities"].values.tolist()

    # divide our questions into n equal batches and get the entites
    generated_entities_batches = get_entities_batch(
        genre_entities, ner_questions, batch_size
    )
    generated_entities = sum(generated_entities_batches, [])  # nd to 1d list

    for idx, generated_entities in enumerate(tqdm(generated_entities)):
        # creating a new question, fetch the english entities of the question
        num_ner_entities = ner_num_entities[idx]
        new_question_obj = QuestionEntitiesCandidates(
            original_questions[idx], ner_questions[idx]
        )
        new_question_obj.get_entities(generated_entities[: num_ner_entities + 1], "en")

        # get the candidate for the question obj
        candidates = candidates_df.loc[idx]
        candidates = list(candidates.to_numpy())

        new_question_obj.populate_candidates(candidates, label2entity, num_ans)
        questions_entities_candidates.append(new_question_obj)

    return questions_entities_candidates


def get_and_save_subgraphs(
    questions_entities_candidates,
    subgraph_obj,
    num_paths,
    dataset_extension_type,
    save_dir,
):
    """
    fetch and save the subgraphs to all of our questions and its meta info
    """
    subgraph_path = save_dir + "subgraph_segments"
    meta_path = save_dir + "meta_segments"
    Path(subgraph_path).mkdir(parents=True, exist_ok=True)
    Path(meta_path).mkdir(parents=True, exist_ok=True)

    idx = 0
    for question in tqdm(questions_entities_candidates):
        candidates = question.candidate_ids
        entities = question.entity_ids
        question.display()

        for cid, candidate in enumerate(tqdm(candidates)):
            subgraph, pathes = subgraph_obj.get_subgraph(entities, candidate, num_paths)

            # saving the meta info
            with open(Path(meta_path) / f"meta_id_{idx}.json", "w+") as f:
                json.dump(
                    {
                        "idx": idx,
                        "original_question": question.original_question,
                        "ner_question": question.ner_question,
                        "question_entity_id": question.entity_ids,
                        "question_entity_text": question.entity_texts,
                        "question_entity_scores": question.entity_scores,
                        "candidate_id": question.candidate_ids[cid],
                        "candidate_text": question.candidate_texts[cid],
                        "target_id": question.candidate_ids[0],
                        "target_text": question.candidate_texts[0],
                        "shortest_paths": pathes,
                    },
                    f,
                )

            if dataset_extension_type == "json":
                subgraphs_to_json(subgraph, subgraph_path + f"/graph_id_{idx}.json")
            else:
                subgraphs_to_pkl(subgraph, subgraph_path + f"/graph_id_{idx}.pkl")
            idx += 1


def subgraphs_to_pkl(subgraph, file_path):
    """
    write our subgraph to pkl file at specified path
    """
    with open(file_path, "wb+") as f:
        pickle.dump(subgraph, f)


def subgraphs_to_json(subgraph, file_path):
    """
    write our subgraph to json file at specified path
    """
    with open(file_path, "w+") as file_handler:
        json.dump(nx.node_link_data(subgraph), file_handler)


if __name__ == "__main__":
    args = parser.parse_args()

    # get our csv and preprocess the question
    df = prepare_csv(args.model_result_path, args.num_rows)

    # load pkl & model to create genre_entities
    lang_title_wikidata_id, trie = load_pkl(
        args.lang_title_wikidata_id_path, args.marisa_trie_path
    )
    model = load_model(args.pretrained_mgenre_weight_path)
    genre_entities = GENREWikidataEntityesCache(
        model,
        trie,
        lang_title_wikidata_id,
    )

    redirect_cache = WikidataRedirectsCache(cache_dir_path="./cache_store")
    label2entity = WikidataLabelToEntity(
        redirect_cache=redirect_cache,
        cache_dir_path="./cache_store",
        sparql_endpoint="http://localhost:7200/repositories/wikidata",
    )

    # array of QuestionEntitiesCandidates
    questions_entities_candidates = []
    questions_entities_candidates = prepare_questions_entities_candidates(
        df,
        questions_entities_candidates,
        genre_entities,
        label2entity,
        args.batch_size,
        args.num_bad_candidates,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # getting our subgraphs
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()
    subgraph_obj = SubgraphsRetriever(
        entity2label,
        shortest_path,
        edge_between_path=args.edge_between_path,
    )
    get_and_save_subgraphs(
        questions_entities_candidates,
        subgraph_obj,
        args.number_of_pathes,
        args.dataset_extension_type,
        args.save_dir,
    )
    # zipping the final subgraphs file
    shutil.make_archive(
        "./subgraphs_dataset/final_datsetv0",
        "zip",
        "./subgraphs_dataset/dataset_v0/subgraph_segments",
    )
