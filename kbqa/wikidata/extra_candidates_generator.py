#!/usr/bin/env python
# coding: utf-8
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments

from typing import Optional

import numpy as np

from ..caches.base import CacheBase
from ..config import DEFAULT_CACHE_PATH
from ..metrics import recall
from .wikidata_label_to_entity import WikidataLabelToEntity
from .wikidata_redirects import WikidataRedirectsCache
from .wikidata_subgraphs_retriever import SubgraphsRetriever


class ExtraCandidateGenerator(CacheBase):
    """Module for generating extra candidates_list and calculating the recall"""

    def __init__(
        self,
        target_list: list,
        candidates_list: list,
        label2entity: WikidataLabelToEntity,
        subgraph_retriever: SubgraphsRetriever,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
    ) -> None:

        super().__init__(cache_dir_path, "wikidata_entity_1_hope_neighbors.pkl")
        self.cache = {}
        self.load_from_cache()

        self.target_list = target_list
        self.candidates_list = candidates_list

        self.label2entity = label2entity
        self.subgraph_retriever = subgraph_retriever

    def seq2seq_recall(
        self, wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None
    ):
        """Function for calculating the recall"""
        return recall(self.target_list, self.candidates_list, wikidata_redirects_cache)

    def seq2seq_recall_with_1hope(
        self, wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None
    ):
        """Function for calculating the recall with 1-hope neighbours"""
        candidates_with_neighbours = self.get_all_1hope_neighbours()
        return recall(
            self.target_list, candidates_with_neighbours, wikidata_redirects_cache
        )

    def get_neighbours_of_candidate(self, candidate_name):
        """Function for retrieving the closest neighbours of entity (1-hope)"""

        if candidate_name not in self.cache:
            candidate = self.label2entity.get_id(candidate_name)

            neighbours = self.subgraph_retriever.get_edges(candidate)["results"][
                "bindings"
            ]
            neighbours_values = [
                neighbour["label"]["value"] for neighbour in neighbours
            ]
            if neighbours_values != []:
                self.cache[candidate_name] = neighbours_values
                self.save_cache()
            else:
                print(f"Empty list of 1-hope neighbours for {candidate}")
                return neighbours_values

        return self.cache[candidate_name]

    def get_all_1hope_neighbours(self):
        """Function for extending the set of candidates_list via 1-hope neighbours"""

        list_of_extra_candidates = []

        for candidates_list in self.candidates_list:

            new_candidates = [
                self.get_neighbours_of_candidate(candidate)
                for candidate in np.unique(np.array(candidates_list))
            ]
            new_candidates = list(np.unique(sum(new_candidates, [])))

            extended_candidates = list(
                np.unique(sum([list(candidates_list), new_candidates], []))
            )
            list_of_extra_candidates.append(list(filter(None, extended_candidates)))

        return list_of_extra_candidates
