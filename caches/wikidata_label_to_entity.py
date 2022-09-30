# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import time
import logging
import requests
from caches.base import CacheBase


class WikidataLabelToEntity(CacheBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(self, cache_dir_path: str = "./cache_store_entity_id") -> None:
        super().__init__(cache_dir_path, "wikidata_entity_to_id.pkl")
        self.cache = {}
        self.load_from_cache()

    def get_id(self, entity_name):
        if entity_name not in self.cache:
            entity_id = self._request_wikidata(entity_name)
            if entity_id is not None:
                self.cache[entity_name] = entity_id

        return self.cache.get(entity_name)

    def _request_wikidata(self, entity_name):
        url = "https://query.wikidata.org/sparql"
        query = """
        SELECT distinct ?item ?itemLabel ?itemDescription WHERE{
        ?item ?label "<ENTITY_NAME>"@en.
        ?article schema:about ?item .
        ?article schema:inLanguage "en" .
        ?article schema:isPartOf <https://en.wikipedia.org/>.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """.replace(
            "<ENTITY_NAME>", entity_name
        )

        def _try_request(query, url):
            try:
                request = requests.get(
                    url, params={"format": "json", "query": query}, timeout=20
                )
                data = request.json()

                return data["results"]["bindings"][0]["item"]["value"].split("/")[-1]

            except ValueError:
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

            except Exception as general_exception:
                logging.exception(general_exception)
                print(f"ERROR with request query:    {query}")
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

        return _try_request(query, url)
