FROM huggingface/transformers-pytorch-gpu:4.22.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt update && \
    apt install wget

COPY ./ /kbqa_package/ 
RUN pip3 install --upgrade pip && \
    pip3 install /kbqa_package && \
    pip3 install uvicorn fastapi

WORKDIR /workspace/

COPY ./deploy/candidate_selection/*.py /workspace/

CMD ["uvicorn", "main:app"]
