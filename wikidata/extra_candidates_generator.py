#!/usr/bin/env python
# coding: utf-8
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments

import numpy as np
from caches.base import CacheBase
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity


class ExtraCandidateGenerator(CacheBase):
    """Module for generating extra candidates_list and calculating the recall"""

    def __init__(
        self,
        target_list: list,
        candidates_list: list,
        label2entity: WikidataLabelToEntity,
        subgraph_retriever: SubgraphsRetriever,
        cache_dir_path: str = "./cache_store",
    ) -> None:

        super().__init__(cache_dir_path, "wikidata_entity_k_hope_neighbors.pkl")
        self.cache = {"1_hope": {}, "2_hope": {}}
        self.load_from_cache()

        self.target_list = target_list
        self.candidates_list = candidates_list

        self.label2entity = label2entity
        self.subgraph_retriever = subgraph_retriever

    def seq2seq_recall(self):
        """Function for calculating the recall"""
        result = [
            int(self.target_list[i] in self.candidates_list[i])
            for i in range(len(self.target_list))
        ]
        return sum(result) / len(self.target_list)

    def seq2seq_recall_with_1hope(self):
        """Function for calculating the recall with 1-hope neighbours"""
        candidates_with_neighbours = self.get_all_1hope_neighbours()
        result = [
            int(self.target_list[i] in candidates_with_neighbours[i])
            for i in range(len(self.target_list))
        ]
        return sum(result) / len(self.target_list)

    def get_neighbours_of_candidate(self, candidate_name):
        """Function for retrieving the closest neighbours of entity (1-hope)"""

        if candidate_name not in self.cache["1_hope"]:
            candidate = self.label2entity.get_id(candidate_name)

            neighbours = self.subgraph_retriever.get_edges(candidate)["results"][
                "bindings"
            ]
            neighbours_values = [
                neighbour["label"]["value"] for neighbour in neighbours
            ]
            if neighbours_values != []:
                self.cache["1_hope"][candidate_name] = neighbours_values
                self.save_cache()
            else:
                print("Empty list of 1-hope neighbours")
                return neighbours_values

        return self.cache["1_hope"][candidate_name]

    def get_2_hope_neighbours(self, candidate_name):
        """Function for retrieving the 2-hope neighbours of entity"""
        if candidate_name not in self.cache["2_hope"]:
            nodes_1 = self.get_neighbours_of_candidate(candidate_name)
            nodes_2 = np.unique(
                sum(
                    [self.get_neighbours_of_candidate(node_i) for node_i in nodes_1], []
                )
            )
            two_hope_neighbours = list(np.unique([*nodes_1, *nodes_2]))
            if two_hope_neighbours:
                self.cache["2_hope"][candidate_name] = two_hope_neighbours
                self.save_cache()
            else:
                print("Empty list of 2-hope neighbours")
                return two_hope_neighbours
        return self.cache["2_hope"][candidate_name]

    def get_all_1hope_neighbours(self):
        """Function for extending the set of candidates_list via 1-hope neighbours"""

        list_of_extra_candidates = []

        for candidates_list in self.candidates_list:

            new_candidates = [
                self.get_neighbours_of_candidate(candidate)
                for candidate in np.unique(candidates_list)
            ]
            new_candidates = list(np.unique(sum(new_candidates, [])))

            extended_candidates = list(
                np.unique(sum([list(candidates_list), new_candidates], []))
            )
            list_of_extra_candidates.append(list(filter(None, extended_candidates)))

        return list_of_extra_candidates
