#!/usr/bin/env python
# coding: utf-8


import sys
import pandas as pd
import numpy as np
from wikidata.extra_candidates_generator import ExtraCandidateGenerator
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever


def main(path_to_results: "path to pd.DataFrame with targets and seq2seq candidates"):
    df_full = pd.read_csv(path_to_results).astype(str)
    target = df_full.target.to_list()
    candidates_answers = np.array(
        df_full[[col for col in df_full.columns if "answer_" in col]]
    )

    subgraph_retriever = SubgraphsRetriever(entity2label=None, shortest_path=None)
    label2entity = WikidataLabelToEntity(redirect_cache=None)
    extra_candidate_generator = ExtraCandidateGenerator(
        target,
        candidates_answers,
        label2entity=label2entity,
        subgraph_retriever=subgraph_retriever,
    )

    seq2seq_1hope = extra_candidate_generator.seq2seq_recall_with_1hope()
    seq2seq = extra_candidate_generator.seq2seq_recall()

    print(f"seq2seq recall with 1-hope neighbours = {np.round(seq2seq_1hope,4)}")

    print(f"seq2seq recall  = {np.round(seq2seq,4)}")


if __name__ == "__main__":
    # print(sys.argv[0])
    main(sys.argv[1])
