# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=W0703
# pylint: disable=R1710
# pylint: disable=R1705
# pylint: disable=C0116
# pylint: disable=R0913
# pylint: disable=W1203

import logging
import time
from collections import Counter

import requests

from ..config import DEFAULT_CACHE_PATH
from .wikidata_label_to_entity import WikidataLabelToEntity
from .wikidata_entity_to_label import WikidataEntityToLabel

from .base import WikidataBase
from .wikidata_redirects import WikidataRedirectsCache


class WikidataEntityRelationships(WikidataBase):
    """WikidataEntityRelationships - class for request relationships  of any wikidata entities with cahce"""

    def __init__(
        self,
        label2entity: WikidataLabelToEntity,
        redirect_cache: WikidataRedirectsCache,
        entity2label: WikidataEntityToLabel,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_entity_relationships.pkl", sparql_endpoint
        )
        self.cache = {"entity_relationships": {}, "n_entities_same": {}}
        self.load_from_cache()
        self.label2entity = label2entity
        self.entity2label = entity2label
        self.redirect_cache = redirect_cache

    def _create_query_params(self, entity_name, language="en"):
        """_create_query_params - function for creating query for entity's subject relationships"""
        params = {
            "action": "wbgetentities",
            "format": "json",
            "languages": language,
            "ids": self.label2entity.get_id(entity_name),
            "props": "claims",
        }
        return params

    def _request_wikidata(
        self,
        entity_name,
        api_endpoint="https://www.wikidata.org/w/api.php",
        language="en",
    ):
        """_request_wikidata - function for ruqesting entity's subject relationships"""

        def _try_request(entity_name, url, language=language):

            query = self._create_query_params(entity_name, language)
            # api_endpoint = "https://www.wikidata.org/w/api.php"
            request = requests.get(url, params=query, timeout=60)
            data = request.json()
            return data

        return _try_request(entity_name, api_endpoint, language=language)

    def get_entity_relationships(
        self,
        entity_name,
        api_endpoint="https://www.wikidata.org/w/api.php",
        language="en",
    ):
        """get_entity_relationships - function for returning subject relationships"""

        # print(entity_id)
        if entity_name not in self.cache["entity_relationships"]:
            try:
                data = self._request_wikidata(entity_name, api_endpoint, language)
                entity_id = self.label2entity.get_id(entity_name)
                property_list = list(data["entities"][entity_id]["claims"].keys())
                property_information = data["entities"][entity_id]["claims"]
                list_of_relationships = []
                for property_ in property_list:
                    property_j = property_information[property_]
                    for object_ in property_j:
                        try:
                            object_i = object_["mainsnak"]["datavalue"]["value"]["id"]
                            list_of_relationships.append([property_, object_i])
                        except Exception:
                            # logging.exception(object_error)

                            logging.error("No object found")

                if list_of_relationships is not None:
                    self.cache["entity_relationships"][
                        entity_id
                    ] = list_of_relationships
                    self.save_cache()
            except Exception:
                # logging.exception(enitity_relation_error)
                logging.error(
                    f"ERROR with finding relationships for entity {str(entity_name)}"
                )
                return []

        return self.cache["entity_relationships"].get(entity_id)

    def get_most_common_relationship(self, list_of_candidates):
        """get_most_common_relationship - function for returning most common predicate-subject relationship"""
        candidates_relations = []
        list_of_candidates = set(list_of_candidates)
        for candidate in list_of_candidates:
            cand_rels = self.get_entity_relationships(candidate)
            candidates_relations.append(cand_rels)

        candidates_relations_flatten = sum(candidates_relations, [])
        candidates_relations_joined = [
            "-".join(cand) for cand in candidates_relations_flatten
        ]
        # most_common_relation = max(
        # set(candidates_relations_joined), key=candidates_relations_joined.count
        # )
        # return most_common_relation.split("-")
        return candidates_relations_joined

    def get_n_entities_with_same_relationship(self, property_id, object_id, n_similar):
        """get_n_entities_with_same_relationship - function for returning 'n_similar' entities with similar predicate-subject relationship"""

        query = (
            """ PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX p: <http://www.wikidata.org/prop/>
                PREFIX v: <http://www.wikidata.org/prop/statement/>
                PREFIX q: <http://www.wikidata.org/prop/qualifier/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT * WHERE {
                ?subject wdt:<PROPERTY_ID> wd:<OBJECT_ID>  .
                } LIMIT <N_SIMILAR_ENTITIES>
                """.replace(
                "<PROPERTY_ID>", property_id
            )
            .replace("<OBJECT_ID>", object_id)
            .replace("<N_SIMILAR_ENTITIES>", str(n_similar))
        )

        def _try_request(query, url, count=0):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    timeout=60,
                    headers={"Accept": "application/json"},
                )
                data = request.json()

                return data["results"]["bindings"]
            except ValueError:
                if count < 3:
                    logging.info("sleep 2...")
                    logging.error("ValueError")
                    time.sleep(2)
                    return _try_request(query, url, count + 1)
                else:
                    logging.error(
                        f"ValueError with object {object_id} and property {property_id}"
                    )
                    return []
            except Exception:
                logging.error(
                    f"ERROR with object {object_id} and property {property_id}"
                )
                return []

        def find_n_similar(query=query, property_id=property_id, object_id=object_id):
            try:
                data = _try_request(query, self.sparql_endpoint)
                n_entity = [
                    self.entity2label.get_label(tmp["subject"]["value"].split("/")[-1])
                    for tmp in data
                ]
                # n_entity=[data[i]["subjectLabel"]["value"] for i in range(len(data))]
                if n_entity is not None:
                    self.cache["n_entities_same"][
                        "_".join([property_id, object_id])
                    ] = n_entity
                    self.save_cache()
            except Exception:
                logging.error(
                    f"ERROR with finding {n_similar} entities with similar relationships for pair {str('_'.join([property_id,object_id]))}"
                )
                return []

        if "_".join([property_id, object_id]) not in self.cache["n_entities_same"]:
            find_n_similar(query=query, property_id=property_id, object_id=object_id)
        else:
            if (
                len(
                    self.cache["n_entities_same"].get(
                        "_".join([property_id, object_id])
                    )
                )
                < n_similar
            ):
                find_n_similar(
                    query=query, property_id=property_id, object_id=object_id
                )

        return self.cache["n_entities_same"].get("_".join([property_id, object_id]))[
            :n_similar
        ]
        # return [data[i]["subjectLabel"]["value"] for i in range(len(data))]

    def get_extra_entities_matrix(
        self, candidates_matrix: "list[list[str]]", n_similar: "int"
    ):
        new_candidates = []
        for candidates_list in candidates_matrix:
            common_rel = self.get_most_common_relationship(set(candidates_list))
            common_rel = dict(Counter(common_rel))
            common_rel = list(
                dict(sorted(common_rel.items(), key=lambda item: item[1], reverse=True))
            )[:1]
            n_similar_candidates = self.get_n_entities_with_same_relationship(
                common_rel[0], common_rel[1], n_similar
            )
            new_candidates.append(n_similar_candidates)
        return new_candidates
