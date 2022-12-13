FROM python:3.9-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update
RUN apt-get install -y python3 python3-pip python-dev build-essential python3-venv
RUN pip install python-telegram-bot requests

RUN mkdir -p /bot
WORKDIR /bot

COPY ./*.py /bot/

CMD ["python3", "/bot/main.py"]
