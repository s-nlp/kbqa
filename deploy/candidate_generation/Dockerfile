FROM huggingface/transformers-pytorch-gpu:4.22.0

RUN pip3 install --upgrade pip && \
    pip3 install altair gradio

WORKDIR /workspace/kbqa/

COPY ./*.py /workspace/kbqa/
COPY ./*.txt /workspace/kbqa/

CMD ["python3", "candidate_generation.py"]
