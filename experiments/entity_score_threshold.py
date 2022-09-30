# setting the path to be at root for import
import sys
import pandas as pd

sys.path[0] = "/workspace/kbqa/"

from tabulate import tabulate
from caches.wikidata_entity_to_label import WikidataEntityToLabel
from caches.genre import GENREWikidataEntityesCache
from subgraphs_dataset.question_entities_candidate import QuestionEntitiesCandidates
from caches.wikidata_subgraphs_retriever import SubgraphsRetriever
from caches.wikidata_shortest_path import WikidataShortestPathCache
from caches.wikidata_entity_to_label import WikidataEntityToLabel
from caches.wikidata_shortest_path import WikidataShortestPathCache
from caches.wikidata_label_to_entity import (
    WikidataLableToEntity,
)  # pylint: disable=import-error,import-error

# pylint: disable=import-error,import-error
from get_subgraphs_dataset import (
    get_csv,
    load_model,
    load_pkl,
    get_question_candidate,
    get_question_entities,
    preprocess_question,
)
from genre.trie import Trie, MarisaTrie  # pylint: disable=unused-import,import-error
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_index_none(array):
    """
    return all indicies of none elements in array
    """
    none_indices = [i for i, v in enumerate(array) if v is None]
    return none_indices


def create_subgraph_retriever():
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()
    subgraph_obj = SubgraphsRetriever(
        entity2label, shortest_path, edge_between_path=True
    )
    return subgraph_obj


def populate_dict(entity_text, entity_id, entity_score, candidate, shortest_path):
    """
    create a dictionary for the entity (to create a pd dataframe)
    """
    res = {
        "entity_text": entity_text,
        "entity_id": entity_id,
        "entity_score": entity_score,
        "candidate": candidate,
        "shortest_path": shortest_path,
    }
    return res


def classify_shortest_path(ques_en_can, candidate, none_indices):
    """
    classify to see which entity results in a shortest paths
    """
    entity_ids = ques_en_can.entity_ids
    entity_scores = ques_en_can.entity_scores
    entity_texts = ques_en_can.entity_texts

    res = []
    for index, _ in enumerate(entity_ids):
        # no shortest path
        if index in none_indices:
            my_dicc = populate_dict(
                entity_texts[index],
                entity_ids[index],
                entity_scores[index],
                candidate,
                0,
            )
        else:  # do have a shortest path
            my_dicc = populate_dict(
                entity_texts[index],
                entity_ids[index],
                entity_scores[index],
                candidate,
                1,
            )

        res.append(my_dicc)

    return res


if __name__ == "__main__":
    # get our csv and preprocess the question
    df = get_csv(5)
    df["question"] = df["question"].apply(preprocess_question)

    lang_title_wikidata_id, trie = load_pkl()
    model = load_model()

    # array of QuestionEntitiesCandidates
    questions_entities_candidates = []

    # get entities
    questions_entities_candidates = get_question_entities(
        df, questions_entities_candidates, model, trie, lang_title_wikidata_id
    )

    # get candidate
    label2entity = WikidataLableToEntity()
    questions_entities_candidates = get_question_candidate(
        df, questions_entities_candidates, label2entity
    )

    # create the subgraph retriever object
    subgraph_obj = create_subgraph_retriever()

    rows_list = []
    for ques_en_can in questions_entities_candidates:
        entities = ques_en_can.entity_ids
        candidates = ques_en_can.candidate_ids

        for candidate in candidates:
            # getting our shortest path from both side
            entity2candidate = subgraph_obj.get_paths(
                entities, candidate, is_entities2candidate=True
            )
            candidate2entity = subgraph_obj.get_paths(
                entities, candidate, is_entities2candidate=False
            )

            # given the shortest paths from both direction, find the shorter path
            paths = subgraph_obj.get_undirected_shortest_path(
                entity2candidate, candidate2entity
            )
            # get the indices that result in no shortest path
            none_indices = get_index_none(paths)

            # see which one has shortest paths and which one don't
            classified_paths = classify_shortest_path(
                ques_en_can, candidate, none_indices
            )
            rows_list += classified_paths
    # create our dataframe
    df = pd.DataFrame(rows_list)
    print(tabulate(df, headers="keys", tablefmt="psql"))
    df.to_csv("/workspace/kbqa/experiments/result.csv", sep="\t")
