from ..wikidata.utils import request_to_wikidata
from ..config import SPARQL_ENDPOINT
from typing import List, Optional, Union
from enum import Enum


class QuestionCandidateRelationSelectorVersion(Enum):
    """Versions for QuestionCandidateRelationSelection"""

    ONE_HOP_DIRECT_CONNECTIONS = 1
    # TWO_HOP_DIRECT_CONNECTIONS = 2
    INSTANCE_OF_CONNECTIONS = 3


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
        # elif version == "TWO_HOP_DIRECT_CONNECTIONS":
        #     self.version = (
        #         QuestionCandidateRelationSelectorVersion.TWO_HOP_DIRECT_CONNECTIONS
        #     )
        elif version == "INSTANCE_OF_CONNECTIONS":
            self.version = (
                QuestionCandidateRelationSelectorVersion.INSTANCE_OF_CONNECTIONS
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
        # elif (
        #     self.version
        #     is QuestionCandidateRelationSelectorVersion.TWO_HOP_DIRECT_CONNECTIONS
        # ):
        #     return QuestionCandidateRelationSelection.filter_two_hop_direct_connections(
        #         question_entities_ids, candidates_ids
        #     )
        elif (
            self.version
            is QuestionCandidateRelationSelectorVersion.INSTANCE_OF_CONNECTIONS
        ):
            (
                filtered,
                _,
            ) = QuestionCandidateRelationSelection.filter_instance_of_connections(
                question_entities_ids, candidates_ids
            )
            return filtered
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
        return list(dict.fromkeys(final_list))

    # @classmethod
    # def filter_two_hop_direct_connections(
    #     cls, question_entities_ids: List[str], candidates_ids: List[str]
    # ) -> List[str]:
    #     final_list = []
    #     for qentity in question_entities_ids:
    #         for candidate in candidates_ids:
    #             for cor_obj in cls._wikidata_get_corresponded_objects(qentity):
    #                 if (
    #                     candidate not in final_list
    #                     and cls._wikidata_get_count_of_intersections(cor_obj, candidate)
    #                     > 0
    #                 ):
    #                     final_list.append(candidate)
    #                     break
    #     return list(dict.fromkeys(final_list))

    @classmethod
    def filter_instance_of_connections(
        cls, question_entities_ids: List[str], candidates_ids: List[str]
    ) -> List[str]:
        internal_states = {"qentity_cor_objects": {}, "candidate_instances_of": {}}
        final_list = []
        for candidate in candidates_ids:
            candidate_instances_of = cls._wikidata_get_instance_of(candidate)
            internal_states["candidate_instances_of"][
                candidate
            ] = candidate_instances_of

            for qentity in question_entities_ids:
                qentity_cor_objects = (
                    cls._wikidata_get_corresponded_objects_with_instance_of(qentity)
                )
                internal_states["qentity_cor_objects"][qentity] = qentity_cor_objects

                for cor_obj_with_instance_of in qentity_cor_objects:
                    if (
                        cor_obj_with_instance_of["instance_of"]
                        in candidate_instances_of
                    ):
                        final_list.append(candidate)
                        break

            # same_instance_of_objects = cls._wikidata_get_same_instance_of_objects(qentity)
            # for candidate in candidates_ids:
            #     if candidate not in final_list and candidate in same_instance_of_objects:
            #         final_list.append(candidate)
        return list(dict.fromkeys(final_list)), internal_states

    @classmethod
    def _wikidata_get_same_instance_of_objects(cls, entity):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?item WHERE 
        {
            wd:<ENTITY> wdt:P31 ?o .
            ?o ^wdt:P31 ?item
        }
        """.replace(
            "<ENTITY>", entity
        )
        objects = request_to_wikidata(query)
        objects = [val["item"]["value"].split("/")[-1] for val in objects]
        return objects

    @classmethod
    def _wikidata_get_count_of_intersections(cls, entity1, entity2):
        try:
            query = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT (count(distinct ?p) as ?count) WHERE {
                {wd:<E1> ?p wd:<E2>} UNION {wd:<E2> ?p wd:<E1>}
            }
            """.replace(
                "<E1>", entity1
            ).replace(
                "<E2>", entity2
            )

            count = request_to_wikidata(query)
            count = int(count[0]["count"]["value"])
        except Exception as exception:
            print(exception)
            count = 0
        return count

    @classmethod
    def _wikidata_get_corresponded_objects_with_instance_of(cls, entity):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT * WHERE {
            {?item ?property wd:<E1>} UNION {wd:<E1> ?property ?item} .
            ?item wdt:P31 ?instance_of .
        }
        """.replace(
            "<E1>", entity
        )

        cor_objects = request_to_wikidata(query, SPARQL_ENDPOINT)
        cor_objects = [
            {
                "property": val["property"]["value"].split("/")[-1],
                "object": val["item"]["value"].split("/")[-1],
                "instance_of": val["instance_of"]["value"].split("/")[-1],
            }
            for val in cor_objects
        ]

        return cor_objects

    @classmethod
    def _wikidata_get_instance_of(cls, entity):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?instance_of WHERE {
            wd:<ENTITY> wdt:P31 ?instance_of
        }
        """.replace(
            "<ENTITY>", entity
        )

        results = request_to_wikidata(query)
        return [val["instance_of"]["value"].split("/")[-1] for val in results]
