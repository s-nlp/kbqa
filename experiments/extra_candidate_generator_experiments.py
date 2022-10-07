#!/usr/bin/env python
# coding: utf-8
# pylint: disable=wrong-import-position,import-error


import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "..")
from wikidata.extra_candidates_generator import ExtraCandidateGenerator
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever
from wikidata.wikidata_redirects import WikidataRedirectsCache
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from wikidata.wikidata_shortest_path import WikidataShortestPathCache


def main(path_to_results: str):
    """main experiment with generating additional candidates based on seq2seq results and calcuating recall

    Args:
        path_to_results (str): path to CSV filw with targets and seq2seq candidates
    """

    df_full = pd.read_csv(path_to_results).astype(str)
    target = df_full.target.to_list()
    candidates_answers = np.array(
        df_full[[col for col in df_full.columns if "answer_" in col]]
    )

    wikidata_redirects_cache = WikidataRedirectsCache(
        sparql_endpoint="http://dbpedia.org/sparql"
    )
    shortest_path_cache = WikidataShortestPathCache()
    entity2label = WikidataEntityToLabel()
    subgraph_retriever = SubgraphsRetriever(
        entity2label=entity2label, shortest_path=shortest_path_cache
    )
    label2entity = WikidataLabelToEntity(redirect_cache=wikidata_redirects_cache)
    extra_candidate_generator = ExtraCandidateGenerator(
        target,
        candidates_answers,
        label2entity=label2entity,
        subgraph_retriever=subgraph_retriever,
    )

    seq2seq_1hope = extra_candidate_generator.seq2seq_recall_with_1hope()
    seq2seq = extra_candidate_generator.seq2seq_recall()

    print(
        f"seq2seq recall (without redirects) with 1-hope neighbours = {np.round(seq2seq_1hope,4)}"
    )
    print(f"seq2seq recall (without redirects)  = {np.round(seq2seq,4)}")

    seq2seq_1hope_redirects = extra_candidate_generator.seq2seq_recall_with_1hope(
        wikidata_redirects_cache
    )
    seq2seq_redirects = extra_candidate_generator.seq2seq_recall(
        wikidata_redirects_cache
    )

    print(
        f"seq2seq recall (with redirects) with 1-hope neighbours = {np.round(seq2seq_1hope_redirects,4)}"
    )
    print(f"seq2seq recall (with redirects)  = {np.round(seq2seq_redirects,4)}")


if __name__ == "__main__":
    # print(sys.argv[0])
    main(sys.argv[1])
