import pandas as pd
import pickle
import torch
import argparse
from genre.fairseq_model import mGENRE  # pylint: disable=import-error
from caches.wikidata_entity_to_label import WikidataEntityToLabel
from caches.genre import GENREWikidataEntityesCache
from subgraphs_dataset.question_entities_candidate import QuestionEntitiesCandidates
from caches.wikidata_subgraphs_retriever import SubgraphsRetriever
from caches.wikidata_shortest_path import WikidataShortestPathCache
from caches.wikidata_label_to_entity import WikidataLableToEntity
from genre.trie import Trie, MarisaTrie  # pylint: disable=unused-import
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_bad_candidates",
    default=5,
    type=int,
)


def get_csv(num_ans: int):
    """
    return the pd df with num_ans answers to our question
    """
    df = pd.read_csv("./results.csv")
    df = df.iloc[:, : num_ans + 3]  # only selecting 50 bad answers
    return df.head(1)


def preprocess_question(question):
    """
    put start and end in our question for mgenre
    """
    question = "[START] " + question.strip() + " [END]"
    return question


def load_pkl():
    """
    load the needed pkl files
    """
    with open(
        "/workspace/kbqa/lang_title2wikidataID-normalized_with_redirect.pkl", "rb"
    ) as f:
        lang_title_wikidata_id = pickle.load(f)

    with open(
        "/workspace/kbqa/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb"
    ) as f:
        trie = pickle.load(f)

    print("finished loading pkl")
    return lang_title_wikidata_id, trie


def load_model():
    """
    load our mGENRE
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mGENRE.from_pretrained(
        "/workspace/kbqa/fairseq_multilingual_entity_disambiguation"
    ).eval()
    model.to(device)
    print("mGENRE loaded")
    return model


def get_question_entities(
    df, questions_entities_candidates, model, trie, lang_title_wikidata_id
):
    """
    get entities to all of our questions
    """
    genre_entities = GENREWikidataEntityesCache(
        model,
        trie,
        lang_title_wikidata_id,
    )
    questions_arr = list(df["question"].to_numpy())
    generated_entities_batched = genre_entities.sentences_batch_to_entities(
        questions_arr
    )
    for idx, generated_entities in enumerate(generated_entities_batched):
        # creating a new question, fetch the english entities of the question
        new_question = QuestionEntitiesCandidates(df["question"][idx])
        new_question.get_entities(generated_entities, "en")
        questions_entities_candidates.append(new_question)

    return questions_entities_candidates


def get_question_candidate(df, questions_entities_candidates, label2entity):
    """
    get the candidates of all our questions
    """
    question_col = df.columns[:1]
    candidates_df = df.drop(columns=question_col)  # dropping our questions
    num_rows = len(candidates_df.index)

    for row_idx in range(num_rows):
        candidates = candidates_df.loc[row_idx]
        candidates = list(candidates.to_numpy())
        questions_entities_candidates[row_idx].populate_candidates(
            candidates, label2entity
        )
        questions_entities_candidates[row_idx].display()

    return questions_entities_candidates


def get_subgraphs(questions_entities_candidates, subgraph_obj):
    """
    fetch the subgraphs to all of our questions
    """
    subgraphs = []
    for question in questions_entities_candidates:
        candidates = question.candidate_ids
        entities = question.entity_ids

        for candidate in candidates:
            subgraph = subgraph_obj.get_subgraph(entities, candidate)
            print(subgraph)
            subgraphs.append(subgraph)

    return subgraphs


def subgraphs_to_pkl(subgraphs, file_path):
    """
    write our subgraph to pkl file at specified path
    """
    with open(file_path, "wb") as f:
        pickle.dump(subgraphs, f)


def fetch_subgraphs_from_pkl(file_path):
    """
    retrieving our subgraphs from pkl file
    """
    with open(file_path, "rb") as f:
        subgraphs = pickle.load(f)
    return subgraphs


if __name__ == "__main__":
    args = parser.parse_args()

    # get our csv and preprocess the question
    df = get_csv(args.num_bad_candidates)
    df["question"] = df["question"].apply(preprocess_question)

    lang_title_wikidata_id, trie = load_pkl()
    model = load_model()

    # array of QuestionEntitiesCandidates
    questions_entities_candidates = []

    questions_entities_candidates = get_question_entities(
        df, questions_entities_candidates, model, trie, lang_title_wikidata_id
    )

    label2entity = WikidataLableToEntity()
    questions_entities_candidates = get_question_candidate(
        df, questions_entities_candidates, label2entity
    )

    # getting our subgraphs and writing it to the pkl files
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()

    subgraph_obj = SubgraphsRetriever(
        entity2label, shortest_path, edge_between_path=True
    )
    edge_between_path_true = get_subgraphs(questions_entities_candidates, subgraph_obj)
    subgraphs_to_pkl(
        edge_between_path_true,
        "/workspace/kbqa/subgraphs_dataset/subgraphs_edges_between.pkl",
    )

    subgraph_obj = SubgraphsRetriever(
        entity2label, shortest_path, edge_between_path=False
    )
    edge_between_path_false = get_subgraphs(questions_entities_candidates, subgraph_obj)
    subgraphs_to_pkl(
        edge_between_path_false,
        "/workspace/kbqa/subgraphs_dataset/subgraphs_no_edges_between.pkl",
    )
