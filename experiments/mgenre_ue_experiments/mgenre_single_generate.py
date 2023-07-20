"""
Generates results of entity linking using single mGENRE model for WDSQ, RuBQ 1-hop
and full RuBQ test sets.
"""

import sys
import pickle

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import stanza  # pylint: disable=import-error
from GENRE.genre.trie import Trie, MarisaTrie  # pylint: disable=import-error
from GENRE.genre.fairseq_model import mGENRE  # pylint: disable=import-error
from spacy_ft.ner_to_sentence_insertion import NerToSentenceInsertion

MASTER_SEED = 42

with open("lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

with open("titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)


def text_to_id(x):
    """Converts predicted sequence to wikidata entity id"""
    splits = x.split(" >> ")
    if len(splits) > 1:
        return max(
            lang_title2wikidataID[tuple(reversed(splits))], key=lambda y: int(y[1:])
        )
    return "Empty"


def predict_sentence(model, sentence):
    """Inference step of mGENRE model"""
    result = model.sample(
        beam=10,
        sentences=[sentence],
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
        ],
        text_to_id=text_to_id,
        marginalize=True,
    )

    for res in result[0]:
        res["scores"] = res["scores"].detach().cpu()
        res["score"] = res["score"].detach().cpu()

    return result[0]


def stanza_nlp(text, nlp=None):
    """Runs Stanza NER on provided text piece"""
    doc = nlp(text)
    return [ent.text for sent in doc.sentences for ent in sent.ents]


def stanza_prediction(sentence, nlp=None):
    """Runs Stanza and places found named entities within start/end tokens"""
    res = stanza_nlp(text=sentence, nlp=nlp)
    if res != []:
        if len(res) == 1:
            first_part, second_part = (
                sentence.split(res[0])[0],
                sentence.split(res[0])[1],
            )
            output = first_part + "[START] " + res[0] + " [END]" + second_part
        else:
            for _ in range(len(res)):
                output = " ".join(
                    [
                        f"[START] {x} [END]" if x in res else x
                        for x in sentence.split(" ")
                    ]
                )
        return output
    return sentence


def get_models():
    """Instantiate models"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation").eval()
    model.to(device)

    ner_stanza = stanza.Pipeline(
        lang="en", processors="tokenize,ner", verbose=False, use_gpu=False
    )
    ner_spacy = NerToSentenceInsertion(model_path="./data/model-best/")

    return model, ner_stanza, ner_spacy


def generate(ner=None):
    """Generates EL results for RuBQ 1-hop and WDSQ datasets"""
    rubq_questions = list(np.load("./data/all_EN_rubq_test_questions_1_hop_uri.npy"))

    sq_test = np.load("./data/simple_questions_test.npy")
    sq_entities = []
    sq_questions = []
    for (entity, _, _, question) in sq_test:
        sq_entities.append(entity)
        sq_questions.append(question)

    model, ner_stanza, ner_spacy = get_models()

    sq_results = []
    rubq_results = []

    for question in tqdm(sq_questions):
        if ner == "stanza":
            question = stanza_prediction(question, nlp=ner_stanza)
        elif ner == "spacy":
            question = ner_spacy.entity_labeling(str(question))
        print(question)
        sq_results.append(predict_sentence(model, question))

    with open(f"results_sq_bs_10_{ner}.pickle", "wb") as handle:
        pickle.dump(sq_results, handle)

    for question in tqdm(rubq_questions):
        if ner == "stanza":
            question = stanza_prediction(question, nlp=ner_stanza)
        elif ner == "spacy":
            question = ner_spacy.entity_labeling(str(question))
        print(question)
        rubq_results.append(predict_sentence(model, question))

    with open(f"results_rubq_bs_10_{ner}.pickle", "wb") as handle:
        pickle.dump(rubq_results, handle)


def generate_full_rubq(ner=None):
    """Generates EL results for full RuBQ test set"""
    rubq_questions = pd.read_json("RuBQ_2.0_test.json")["question_eng"]

    model, ner_stanza, ner_spacy = get_models()

    rubq_results = []

    for question in tqdm(rubq_questions[0:5]):
        if ner == "stanza":
            question = stanza_prediction(question, nlp=ner_stanza)
        elif ner == "spacy":
            question = ner_spacy.entity_labeling(str(question))
        print(question)
        rubq_results.append(predict_sentence(model, question))

    with open(f"results_full_rubq_bs_10_{ner}.pickle", "wb") as handle:
        pickle.dump(rubq_results, handle)


def main(ner):
    """Entry point for result generation"""
    generate(ner=ner)
    generate_full_rubq(ner=ner)


if __name__ == "__main__":
    print("Using NER: ", sys.argv[1])
    main(sys.argv[1])
