import pandas as pd
import pickle
import torch
import argparse
import json
from tqdm import tqdm

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
    default="pkl",
    type=str,
    choices=["json", "pkl"],
)
parser.add_argument(
    "--batch_size",
    default=50,
    type=int,
)
parser.add_argument("--save_dir", default="./subgraphs_dataset/dataset_v0/", type=str)


def prepare_csv(path):
    """
    return the pd df with num_ans answers to our question
    """
    df = pd.read_csv(path)
    print("loaded results.csv")
    return df.head(100)


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
    file_path,
):
    """
    prepare all questions, question entities and candidates for subgraph generation
    """
    original_questions = df["question"].values.tolist()
    ner = NerToSentenceInsertion(model_path=args.ner_model_path)
    df["question"] = df["question"].apply(ner.entity_labeling)
    ner_questions = df[
        "question"
    ].values.tolist()  # getting questions with START and END
    candidates_df = df.drop(columns=df.columns[:1])  # dropping questions col

    # divide our questions into n equal batches and get the entites
    generated_entities_batches = get_entities_batch(
        genre_entities, ner_questions, batch_size
    )
    generated_entities = sum(generated_entities_batches, [])  # nd to 1d list

    for idx, generated_entities in enumerate(tqdm(generated_entities)):
        # creating a new question, fetch the english entities of the question
        new_question_obj = QuestionEntitiesCandidates(
            original_questions[idx], ner_questions[idx]
        )
        new_question_obj.get_entities(generated_entities, "en")

        # get the candidate for the question obj
        candidates = candidates_df.loc[idx]
        candidates = list(candidates.to_numpy())

        new_question_obj.populate_candidates(candidates, label2entity, num_ans)
        questions_entities_candidates.append(new_question_obj)

    # save meta info after processing
    save_meta_info(questions_entities_candidates, file_path)
    return questions_entities_candidates


def get_subgraphs(questions_entities_candidates, subgraph_obj, num_paths):
    """
    fetch the subgraphs to all of our questions
    """
    subgraphs = []
    for question in tqdm(questions_entities_candidates):
        candidates = question.candidate_ids
        entities = question.entity_ids
        question.display()

        for candidate in tqdm(candidates):
            subgraph = subgraph_obj.get_subgraph(entities, candidate, num_paths)
            subgraphs.append(subgraph)
    return subgraphs


def subgraphs_to_pkl(subgraphs, file_path):
    """
    write our subgraph to pkl file at specified path
    """
    with open(file_path, "wb") as f:
        pickle.dump(subgraphs, f)


def subgraphs_to_json(subgraphs, file_path):
    """
    write our subgraph to json file at specified path
    """
    for subgraph in subgraphs:
        with open(file_path, "w") as file_handler:
            json.dump(nx.node_link_data(subgraph), file_handler)


def save_meta_info(questions_entities_candidates, file_path):
    """
    save meta info to json file at specified path
    """
    res_dict = []
    for idx, question_obj in enumerate(questions_entities_candidates):
        res_dict.append(
            {
                "idx": idx,
                "original_question": question_obj.original_question,
                "ner_question": question_obj.ner_question,
                "question_entity_id": question_obj.entity_ids,
                "question_entity_text": question_obj.entity_texts,
                "question_entity_scores": question_obj.entity_scores,
                "candidate_id": question_obj.candidate_ids[1:],
                "candidate_text": question_obj.candidate_texts[1:],
                "target_id": question_obj.candidate_ids[0],
                "target_text": question_obj.candidate_texts[0],
            }
        )
    with open(file_path + "meta.json", "w") as file_handler:
        json.dump(
            res_dict,
            file_handler,
        )


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
    df = prepare_csv(args.model_result_path)

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
        args.save_dir,
    )

    # getting our subgraphs
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()
    subgraph_obj = SubgraphsRetriever(
        entity2label,
        shortest_path,
        edge_between_path=args.edge_between_path,
    )
    subgraphs = get_subgraphs(
        questions_entities_candidates,
        subgraph_obj,
        args.number_of_pathes,
    )

    # saving the final subgraphs
    if args.dataset_extension_type == "pkl":
        subgraphs_to_pkl(
            subgraphs,
            args.save_dir
            + "/subgraphs_edges_between_{}.pkl".format(str(args.edge_between_path)),
        )
    else:
        subgraphs_to_json(
            subgraphs,
            args.save_dir
            + "/subgraphs_edges_between_{}.json".format(str(args.edge_between_path)),
        )
