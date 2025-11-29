FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y git htop g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY ./requirements.txt /
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt

COPY ./ /workspace/kbqa
RUN pip install -e /workspace/kbqa

WORKDIR /workspace/kbqa

