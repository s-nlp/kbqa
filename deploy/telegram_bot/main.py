# pylint: disable=unused-argument

import os
import requests
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
)
from telegram import ChatAction
from label_to_entity import LabelToEntity
import requests
import time
import os
from functools import lru_cache
import logging


CANDIDATE_GENERATION_URI = os.getenv(
    "CANDIDATE_GENERATION_URI", "http://localhost:7860"
)

ENTITY_LINKING_URI = os.getenv("ENTITY_LINKING_URI", "http://localhost:7861")
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def get_module_logger(mod_name):
    """
    To use this, do logger = get_module_logger(__name__)
    """
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_module_logger("tg_bot")


@lru_cache(maxsize=16384)
def wikidata_request(query):
    logger.info(f"Request to Wikidata with query:\n{query}")
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"format": "json", "query": query},
        headers={"Accept": "application/json"},
    )
    to_sleep = 0.2
    while response.status_code == 429:
        if "retry-after" in response.headers:
            to_sleep += int(response.headers["retry-after"])
        to_sleep += 0.5
        logger.info(f"wikidata_request to sleep...")
        time.sleep(to_sleep)
        response = requests.get(
            SPARQL_ENDPOINT,
            params={"format": "json", "query": query},
            headers={"Accept": "application/json"},
        )
    return response.json()["results"]["bindings"]


@lru_cache(maxsize=16384)
def wikidata_get_count_of_intersections(entity1, entity2):
    query = """
SELECT (count(distinct ?p) as ?count) WHERE {
  {wd:<E1> ?p wd:<E2>} UNION {wd:<E2> ?p wd:<E1>}
}
    """.replace(
        "<E1>", entity1
    ).replace(
        "<E2>", entity2
    )

    count = wikidata_request(query)
    return int(count[0]["count"]["value"])


def wikidata_get_corresponded_objects(entity):
    query = """
SELECT ?p ?item WHERE {
    wd:<E1> ?p ?item .
    ?article schema:about ?item .
    ?article schema:inLanguage "en" .
    ?article schema:isPartOf <https://en.wikipedia.org/> .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
    """.replace(
        "<E1>", entity
    )

    cor_objects = wikidata_request(query)
    cor_objects = [val["item"]["value"].split("/")[-1] for val in cor_objects]

    return cor_objects


def filter_candidates(candidate_ids, q_entity_ids):
    final_list = []
    for qentity in q_entity_ids:
        for candidate in candidate_ids:
            if wikidata_get_count_of_intersections(qentity, candidate) > 0:
                final_list.append(candidate)
                logger.info(f"{qentity} and {candidate} has direct path")
                break
    return list(dict.fromkeys(final_list))


def filter_candidates_second_order(candidate_ids, q_entity_ids):
    final_list = []
    for qentity in q_entity_ids:
        for candidate in candidate_ids:
            for cor_obj in wikidata_get_corresponded_objects(qentity):
                if wikidata_get_count_of_intersections(cor_obj, candidate) > 0:
                    final_list.append(candidate)
                    logger.info(f"{qentity} and {candidate} has direct path")
                    break
        return list(dict.fromkeys(final_list))


logger.info("Loading LabelToEntity start...")
label2entity = LabelToEntity()
logger.info("Loading LabelToEntity done")


def hello(update: Update, context: CallbackContext) -> None:
    intro_text = f""" \n
👋 Greetings {update.effective_user.first_name}! \n
🤖 I'm a bot for answer your questions.
🦾 I can generate set of answers candidates sorted by uncertanity.
⏱️ Please be patient, it may take more than 60 seconds depending on the load.
    """
    update.message.reply_text(intro_text)


def get_candidate_generation_response(question):
    answer_candidates_responce = requests.post(
        f"{CANDIDATE_GENERATION_URI}/run/predict",
        json={
            "data": [
                question,
            ]
        },
    ).json()
    logger.info("answer_candidates_responce: " + str(answer_candidates_responce))
    answer_candidates_list = answer_candidates_responce["data"][0]
    return answer_candidates_list


def get_entity_linking_response(question):
    entity_linker_responce = requests.post(
        f"{ENTITY_LINKING_URI}/run/predict",
        json={
            "data": [
                question,
            ]
        },
    ).json()
    logger.info("entity_linker_responce: " + str(entity_linker_responce))
    entity_linked_results = entity_linker_responce["data"][0]
    return entity_linked_results


def get_entity_id_by_label(label):
    label = str(label)
    if " >> " not in label:
        label += " >> en"

    return label2entity.text_to_id(label)


def respond_to_user(update: Update, context: CallbackContext):
    question = update.message.text

    update.message.chat.send_action(action=ChatAction.TYPING)
    answer_candidates_list = get_candidate_generation_response(question)
    answer_candidates_ids = [get_entity_id_by_label(a) for a in answer_candidates_list]
    answer_candidates_ids = [a for a in answer_candidates_ids if a is not None]
    answer_candidates_text = "\n".join(
        [
            str(a) + " (" + str(get_entity_id_by_label(a)) + ")"
            for a in answer_candidates_list
            if get_entity_id_by_label(a) is not None
        ]
    )
    update.message.reply_text(
        f"ANSWER CANDIDATES for Question:\n** {question} **:\n\n"
        + answer_candidates_text
    )

    update.message.chat.send_action(action=ChatAction.TYPING)
    entity_linked_results = get_entity_linking_response(question)
    entity_linked_ids = [
        get_entity_id_by_label(a)
        for a in entity_linked_results["final_linked_entities_list"]
    ]
    entity_linked_ids = [a for a in entity_linked_ids if a is not None]
    final_linked_entities_text = "\n".join(
        [
            str(e) + " (" + str(get_entity_id_by_label(e)) + ")"
            for e in entity_linked_results["final_linked_entities_list"]
        ]
    )
    update.message.reply_text(
        f"LINKED ENTITIES for Question:\n** {question} **:\n\n"
        + final_linked_entities_text
    )

    update.message.chat.send_action(action=ChatAction.TYPING)
    filtered_candidates = filter_candidates(answer_candidates_ids, entity_linked_ids)
    final_answer_text = str(filtered_candidates)
    update.message.reply_text(
        f"FINAL ANSWER First Order for Question:\n** {question} **:\n\n"
        + final_answer_text
    )

    update.message.chat.send_action(action=ChatAction.TYPING)
    filtered_candidates = filter_candidates_second_order(
        answer_candidates_ids, entity_linked_ids
    )
    final_answer_text = str(filtered_candidates)
    update.message.reply_text(
        f"FINAL ANSWER Second Order Selection for Question:\n** {question} **:\n\n"
        + final_answer_text
    )


if __name__ == "__main__":
    if os.environ.get("TELEGRAM_BOT_KEY") is None:
        raise Exception(
            "Provide TELEGRAM_BOT_KEY environ for using bot with KEY for your bot"
        )

    updater = Updater(os.environ["TELEGRAM_BOT_KEY"])
    updater.dispatcher.add_handler(CommandHandler("start", hello))
    updater.dispatcher.add_handler(
        MessageHandler(Filters.text & ~Filters.command, respond_to_user)
    )
    updater.start_polling()
    updater.idle()
