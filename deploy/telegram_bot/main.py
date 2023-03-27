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
import os
import logging


CANDIDATE_GENERATION_URI = os.getenv(
    "CANDIDATE_GENERATION_URI", "http://localhost:7860"
)
ENTITY_LINKING_URI = os.getenv("ENTITY_LINKING_URI", "http://localhost:7861")
CANDIDATE_SELECTION_URI = os.getenv("CANDIDATE_SELECTION_URI", "http://localhost:7862")
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


def get_direct_connection_selection(answer_candidates_ids, entity_linked_ids, version):
    if version == "one_hop":
        uri = (
            f"{CANDIDATE_SELECTION_URI}/relation_selection/one_hop_direct_connections/"
        )
    else:
        uri = (
            f"{CANDIDATE_SELECTION_URI}/relation_selection/two_hop_direct_connections/"
        )

    responce = requests.get(
        uri,
        json={
            "question_entities_ids": entity_linked_ids,
            "candidates_ids": answer_candidates_ids,
        },
        headers={
            "Content-Type": "application/json",
        },
    ).json()
    logger.info("get_direct_connection_selection_responce: " + str(responce))
    return responce


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
    filtered_candidates = get_direct_connection_selection(
        answer_candidates_ids, entity_linked_ids, "one_hop"
    )
    final_answer_text = str(filtered_candidates)
    update.message.reply_text(
        f"FINAL ANSWER First Order for Question:\n** {question} **:\n\n"
        + final_answer_text
    )

    update.message.chat.send_action(action=ChatAction.TYPING)
    filtered_candidates = get_direct_connection_selection(
        answer_candidates_ids, entity_linked_ids, "two_hop"
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
