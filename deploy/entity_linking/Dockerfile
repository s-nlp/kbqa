FROM huggingface/transformers-pytorch-gpu:4.22.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt update && \
    apt install wget

RUN pip3 install --upgrade pip && \
    pip3 install spacy==3.4 altair gradio nltk numpy

WORKDIR /workspace/

COPY ./src/ /workspace/src/
COPY ./*.py /workspace/
COPY ./*.txt /workspace/

CMD ["python3", "./main.py"]
