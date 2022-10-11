# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=W0703


import time
import requests
from wikidata.wikidata_label_to_entity import WikidataLabelToEntity
from wikidata.base import WikidataBase
from wikidata.wikidata_redirects import WikidataRedirectsCache


class WikidataEntityRelationships(WikidataBase):
    """WikidataEntityRelationships - class for request relationships  of any wikidata entities with cahce"""

    def __init__(
        self,
        label2entity: WikidataLabelToEntity,
        redirect_cache: WikidataRedirectsCache,
        cache_dir_path: str = "./cache_store",
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_entity_relationships.pkl", sparql_endpoint
        )
        self.cache = {}
        self.load_from_cache()
        self.label2entity = label2entity
        self.redirect_cache = redirect_cache

    def _create_query_params(self, entity_id, language="en"):
        """_create_query_params - function for creating query for entity's subject relationships"""
        params = {
            "action": "wbgetentities",
            "format": "json",
            "languages": language,
            "ids": entity_id,
            "props": "claims",
        }
        return params

    def _request_wikidata(
        self,
        entity_id,
        api_endpoint="https://www.wikidata.org/w/api.php",
        language="en",
    ):
        """_request_wikidata - function for ruqesting entity's subject relationships"""
        query = self._create_query_params(entity_id, language)
        ###api_endpoint = "https://www.wikidata.org/w/api.php"

        def _try_request(query, url):
            try:

                request = requests.get(url, params=query, timeout=60)
                data = request.json()

                return data

            except ValueError:
                print("sleep 60...")
                # time.sleep(60)
                return _try_request(query, url)

            except Exception:
                # logging.exception(request_exception)
                print(f"ERROR with entity {entity_id}, fetching for redirects")
                redirects = self.redirect_cache.get_redirects(entity_id)

                if redirects == "No results found":
                    return ""
                for redirect in redirects:
                    new_query = self._create_query_params(redirect, language)
                    return _try_request(new_query, url)

        return _try_request(query, api_endpoint)

    def get_entity_relationships(
        self,
        entity_id,
        api_endpoint="https://www.wikidata.org/w/api.php",
        language="en",
    ):
        """get_entity_relationships - function for returning subject relationships"""

        # print(entity_id)
        if entity_id not in self.cache:
            try:
                data = self._request_wikidata(entity_id, api_endpoint, language)
                properties = data["entities"][entity_id]["claims"]
                property_list = list(data["entities"][entity_id]["claims"].keys())
                list_of_relationships = []

                for property_ in property_list:
                    property_j = properties[property_]
                    for object_ in property_j:
                        try:
                            object_i = object_["mainsnak"]["datavalue"]["value"]["id"]
                            list_of_relationships.append([property_, object_i])
                        except Exception:
                            # logging.exception(object_error)

                            print("No object found")

                if entity_id is not None:
                    self.cache[entity_id] = list_of_relationships
            except Exception:
                # logging.exception(enitity_relation_error)
                print("ERROR with finding reltionships for entity {entity_id}")
                return []

        return self.cache.get(entity_id)

    def get_most_common_relationship(self, list_of_candidates):
        """get_most_common_relationship - function for returning most common predicate-subject relationship"""
        candidates_relations = []
        for candidate in list_of_candidates:
            candidates_relations.append(
                self.get_entity_relationships(self.label2entity.get_id(candidate))
            )

        candidates_relations_flatten = sum(candidates_relations, [])
        candidates_relations_joined = [
            "-".join(cand) for cand in candidates_relations_flatten
        ]
        most_common_relation = max(
            set(candidates_relations_joined), key=candidates_relations_joined.count
        )
        return most_common_relation.split("-")

    def get_n_entities_with_same_relationship(
        self, property_id, object_id, n_similar, language="en"
    ):
        """get_n_entities_with_same_relationship - function for returning 'n_similar' entities with similar predicate-subject relationship"""

        query = (
            """
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX p: <http://www.wikidata.org/prop/>
                PREFIX v: <http://www.wikidata.org/prop/statement/>
                PREFIX q: <http://www.wikidata.org/prop/qualifier/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT  ?subjectLabel ?value WHERE {
                  ?subject wdt:<PROPERTY_ID> wd:<OBJECT_ID>  .

                  SERVICE wikibase:label {
                    bd:serviceParam wikibase:language <LANGUAGE> .
                  }
                } LIMIT <N_SIMILAR_ENTITIES>

                """.replace(
                "<PROPERTY_ID>", property_id
            )
            .replace("<OBJECT_ID>", object_id)
            .replace("<N_SIMILAR_ENTITIES>", str(n_similar))
            .replace("<LANGUAGE>", language)
        )

        def _try_request(query, url):
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
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

            except Exception:
                # logging.exception(similar_relations_error)

                print(f"ERROR with object {object_id} and property {property_id}")

                return []

        data = _try_request(query, self.sparql_endpoint)

        return [data[i]["subjectLabel"]["value"] for i in range(len(data))]
