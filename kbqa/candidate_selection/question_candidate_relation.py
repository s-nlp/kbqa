from ..wikidata.utils import request_to_wikidata
from typing import List, Optional, Union
from enum import Enum


class QuestionCandidateRelationSelectorVersion(Enum):
    """Versions for QuestionCandidateRelationSelection"""

    ONE_HOP_DIRECT_CONNECTIONS = 1
    TWO_HOP_DIRECT_CONNECTIONS = 2


class QuestionCandidateRelationSelection:
    """QuestionCandidateRelationSelection - class for select candidates by relations
    versions:
    """

    def __init__(
        self,
        version: Optional[Union[str, QuestionCandidateRelationSelectorVersion]] = None,
    ):
        if isinstance(version, QuestionCandidateRelationSelectorVersion):
            self.version = version
        elif version is None or version == "ONE_HOP_DIRECT_CONNECTIONS":
            self.version = (
                QuestionCandidateRelationSelectorVersion.ONE_HOP_DIRECT_CONNECTIONS
            )
        elif version == "TWO_HOP_DIRECT_CONNECTIONS":
            self.version = (
                QuestionCandidateRelationSelectorVersion.TWO_HOP_DIRECT_CONNECTIONS
            )
        else:
            raise ValueError(f"version {version} not supported")

    def __call__(
        self,
        question_entities_ids: List[str],
        candidates_ids: List[str],
    ) -> List[str]:
        if (
            self.version
            is QuestionCandidateRelationSelectorVersion.ONE_HOP_DIRECT_CONNECTIONS
        ):
            return QuestionCandidateRelationSelection.filter_one_hop_direct_connections(
                question_entities_ids, candidates_ids
            )
        elif (
            self.version
            is QuestionCandidateRelationSelectorVersion.TWO_HOP_DIRECT_CONNECTIONS
        ):
            return QuestionCandidateRelationSelection.filter_two_hop_direct_connections(
                question_entities_ids, candidates_ids
            )
        else:
            raise NotImplementedError(
                f"Candidates selector {self.version} not supported."
            )

    @classmethod
    def filter_one_hop_direct_connections(
        cls, question_entities_ids: List[str], candidates_ids: List[str]
    ) -> List[str]:
        final_list = []
        for qentity in question_entities_ids:
            for candidate in candidates_ids:
                if (
                    candidate not in final_list
                    and cls._wikidata_get_count_of_intersections(qentity, candidate) > 0
                ):
                    final_list.append(candidate)
                    break
        return list(dict.fromkeys(final_list))

    @classmethod
    def filter_two_hop_direct_connections(
        cls, question_entities_ids: List[str], candidates_ids: List[str]
    ) -> List[str]:
        final_list = []
        for qentity in question_entities_ids:
            for candidate in candidates_ids:
                for cor_obj in cls._wikidata_get_corresponded_objects(qentity):
                    if (
                        candidate not in final_list
                        and cls._wikidata_get_count_of_intersections(cor_obj, candidate)
                        > 0
                    ):
                        final_list.append(candidate)
                        break
        return list(dict.fromkeys(final_list))

    @classmethod
    def _wikidata_get_count_of_intersections(cls, entity1, entity2):
        query = """
            SELECT (count(distinct ?p) as ?count) WHERE {
            {wd:<E1> ?p wd:<E2>} UNION {wd:<E2> ?p wd:<E1>}
        }
        """.replace(
            "<E1>", entity1
        ).replace(
            "<E2>", entity2
        )

        count = request_to_wikidata(query)
        return int(count[0]["count"]["value"])

    @classmethod
    def _wikidata_get_corresponded_objects(cls, entity):
        query = """
        SELECT ?p ?item WHERE {
            wd:<E1> ?p ?item .
            ?article schema:about ?item .
            ?article schema:inLanguage "en" .
            ?article schema:isPartOf <https://en.wikipedia.org/> .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """.replace(
            "<E1>", entity
        )

        cor_objects = request_to_wikidata(query)
        cor_objects = [val["item"]["value"].split("/")[-1] for val in cor_objects]

        return cor_objects
