""" generating sugbraph dataset script"""
from pathlib import Path
import pickle
import argparse
import json
import jsonlines
from tqdm import tqdm
import pandas as pd

from kbqa.wikidata import (
    WikidataEntityToLabel,
    WikidataRedirectsCache,
    SubgraphsRetrieverIgraph,
    WikidataShortestPathIgraphCache,
)
from question_entities_candidate import QuestionEntitiesCandidates

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_bad_candidates",
    default=3,
    type=int,
)
parser.add_argument(
    "--igraph_wikidata_path",
    default="/workspace/storage/wikidata_igraph_v2",
    type=str,
)
parser.add_argument(
    "--edge_between_path",
    default=True,
    type=bool,
)
parser.add_argument(
    "--number_of_pathes",
    default=10,
    type=int,
    help="maximum number of shortest pathes that will queried from KG for entities pair",
)
parser.add_argument(
    "--model_result_path",
    default="./subgraphs_dataset/results.csv",
    type=str,
)
# pylint: disable=line-too-long
parser.add_argument(
    "--sqwd_jsonl_path",
    default="/workspace/storage/sqwd_to_subgraphs_prepared_entities/sqwd_train.jsonl",
    type=str,
)

parser.add_argument(
    "--save_dir",
    default="./subgraphs_dataset/dataset_v3_TRAIN/",
    type=str,
)


def prepare_csv(path):
    """
    return the pd df with num_ans answers to our question
    """
    res = pd.read_csv(path)
    res.drop("target_out_of_vocab", axis=1, inplace=True)
    res["target"] = res["target"].str.strip("['']")
    print("loaded results.csv")
    return res


def parse_jsonl(reader_list, question, max_len):
    """
    parse the jsonl reader list for the current question
    """
    candidates = []
    question_entity = []
    gold_query = None

    for idx, curr_line in enumerate(reader_list):
        if curr_line["question"].strip() == question.strip():
            if gold_query is None:
                gold_query = curr_line["groundTruthAnswerEntity"]
            if curr_line["questionEntity"][0] not in question_entity:
                question_entity.extend(curr_line["questionEntity"])
            if curr_line["answerEntity"][0] not in candidates:
                candidates.extend(curr_line["answerEntity"])

            # if next line is a different question, break
            if idx < max_len:
                if (reader_list[idx + 1]["question"]).strip() != question.strip():
                    break
    return candidates, question_entity, gold_query


def prepare_questions_entities_candidates(
    res_df,
    sqwd_json_path,
    entity2label,
    num_ans,
):
    """
    prepare all questions, question entities and candidates for subgraph generation
    """
    questions_entities_candidates = []
    questions_arr = list(res_df["question"])
    jsonl_reader = jsonlines.open(sqwd_json_path)
    jsonl_reader_list = list(jsonl_reader)
    jsonl_reader.close()

    for curr_question in tqdm(questions_arr):
        try:
            candidates, question_entity, gold_query = parse_jsonl(
                jsonl_reader_list, curr_question, len(questions_arr) - 1
            )
            if len(candidates) == 0 or len(question_entity) == 0 or gold_query is None:
                continue
            new_question_obj = QuestionEntitiesCandidates(curr_question.strip())
            new_question_obj.entity_ids = question_entity
            new_question_obj.entity_texts = [
                entity2label.get_label(entity) for entity in question_entity
            ]

            # get the candidate for the question obj
            candidates = gold_query + candidates
            new_question_obj.populate_candidates(candidates, entity2label, num_ans)
            questions_entities_candidates.append(new_question_obj)
        except KeyboardInterrupt:
            print("exiting from parsing for questions entities candidates")
            break

    return questions_entities_candidates


# pylint: disable=too-many-locals
def get_and_save_subgraphs(
    questions_entities_candidates,
    subgraph_obj,
    num_paths,
    save_dir,
):
    """
    fetch and save the subgraphs to all of our questions and its meta info
    """
    subgraph_path = save_dir + "subgraph_segments"
    meta_path = save_dir + "meta_segments"
    Path(subgraph_path).mkdir(parents=True, exist_ok=True)
    Path(meta_path).mkdir(parents=True, exist_ok=True)

    for qid, question in enumerate(tqdm(questions_entities_candidates)):
        candidates = question.candidate_ids
        # if our gold query doesn't start with Q
        if not candidates[0].startswith("Q"):
            continue
        entities = question.entity_ids
        question.display()

        for cid, candidate in enumerate(candidates):
            try:
                if not candidate.startswith("Q"):
                    continue
                print(
                    f"{cid}/{len(candidates)} \t getting subgraph for {entities} -> {candidate}"
                )
                subgraph, pathes = subgraph_obj.get_subgraph(
                    entities, candidate, num_paths
                )
                # saving the meta info
                with open(
                    Path(meta_path) / f"meta_id_{qid}.json", "w+", encoding="utf-8"
                ) as json_f:
                    json.dump(
                        {
                            "idx": qid,
                            "original_question": question.original_question,
                            "question_entity_id": question.entity_ids,
                            "question_entity_text": question.entity_texts,
                            "question_entity_scores": question.entity_scores,
                            "candidate_id": question.candidate_ids[cid],
                            "candidate_text": question.candidate_texts[cid],
                            "target_id": question.candidate_ids[0],
                            "target_text": question.candidate_texts[0],
                            "shortest_paths": pathes,
                        },
                        json_f,
                    )
                # making a folder for each question
                question_graphs_path = subgraph_path + f"/question_{qid}"
                Path(question_graphs_path).mkdir(parents=True, exist_ok=True)
                # saving our graph under each question
                classifer = (
                    "correct" if candidate == question.candidate_ids[0] else "wrong"
                )
                with open(
                    question_graphs_path + f"/graph_{classifer}_{cid}.pkl", "wb+"
                ) as handle:
                    pickle.dump(subgraph, handle)
                # break
            except KeyboardInterrupt:
                print("exiting from subgraphs retrieval")


if __name__ == "__main__":
    args = parser.parse_args()

    # get our csv and preprocess the question
    result_df = prepare_csv(args.model_result_path)
    redirect_cache = WikidataRedirectsCache(cache_dir_path="./cache_store")
    entity2label_obj = WikidataEntityToLabel()

    # getting array of QuestionEntitiesCandidates
    ques_ent_candidates_obj = prepare_questions_entities_candidates(
        result_df,
        args.sqwd_jsonl_path,
        entity2label_obj,
        args.num_bad_candidates,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data_dir = args.igraph_wikidata_path
    graph_path = data_dir + "/wikidata_triples.txt"

    # getting our subgraphs
    shortest_path = WikidataShortestPathIgraphCache(graph_path=graph_path)
    subgraph_object = SubgraphsRetrieverIgraph(
        entity2label_obj, shortest_path, edge_between_path=False
    )

    get_and_save_subgraphs(
        ques_ent_candidates_obj,
        subgraph_object,
        args.number_of_pathes,
        args.save_dir,
    )
