FROM huggingface/transformers-pytorch-gpu:4.29.2

RUN apt update && \
    apt install -y git htop g++ && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/g++ 10

COPY ./requirements.txt /
RUN pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt

RUN git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/MihailSalnikov/fairseq && \
    cd /fairseq && \
    pip3 install --editable ./ && \
    cd / && \
    echo "export PYTHONPATH=/fairseq/" >> ~/.bashrc

RUN git clone https://github.com/facebookresearch/KILT.git && \
    pip3 install ./KILT

RUN git clone https://github.com/MihailSalnikov/GENRE.git && \
    pip3 install ./GENRE

