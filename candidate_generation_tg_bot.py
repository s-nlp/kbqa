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

if os.environ.get("TG_QA_CG_BOT_KEY") is None:
    raise Exception(
        "Provide TG_QA_CG_BOT_KEY environ for using bot with KEY for your bot"
    )


def hello(update: Update, context: CallbackContext) -> None:
    intro_text = f""" \n
    👋 Greetings {update.effective_user.first_name}! \n
    🤖 I'm a bot for answer your questions.
    🦾 I can generate set of answers candidates sorted by uncertanity.
    ⏱️ Please be patient, it may take more than 30 seconds depending on the load.
    """
    update.message.reply_text(intro_text)


def get_candidate_generation_response(question):
    response = requests.post(
        "http://127.0.0.1:7860/run/predict",
        json={
            "data": [
                question,
            ]
        },
    ).json()

    return response["data"][0]


def respond_to_user(update: Update, context: CallbackContext):
    update.message.chat.send_action(action=ChatAction.TYPING)
    response_list = get_candidate_generation_response(update.message.text)
    response_text = "\n".join(response_list)
    update.message.reply_text(response_text)


updater = Updater(os.environ["TG_QA_CG_BOT_KEY"])
updater.dispatcher.add_handler(CommandHandler("start", hello))
updater.dispatcher.add_handler(
    MessageHandler(Filters.text & ~Filters.command, respond_to_user)
)
updater.start_polling()
updater.idle()
