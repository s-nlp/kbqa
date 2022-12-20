import time

import requests

from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase


class WikidataEntityToLabel(WikidataBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_entity_to_label.pkl", sparql_endpoint
        )
        self.cache = {}
        self.load_from_cache()

    def get_label(self, entity_idx):
        if entity_idx not in self.cache:
            label = self._request_wikidata(entity_idx)
            if label is not None:
                self.cache[entity_idx] = label
                self.save_cache()

        return self.cache.get(entity_idx)

    def _request_wikidata(self, entity_idx):
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX wd: <http://www.wikidata.org/entity/> 
        SELECT  *
        WHERE {
            wd:<ENTITY> rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
        } 
        """.replace(
            "<ENTITY>", entity_idx
        )

        def _try_request(query, url):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    headers={"Accept": "application/json"},
                )
                data = request.json()

                if len(data["results"]["bindings"]) == 0:
                    return None

                return data["results"]["bindings"][0]["label"]["value"]

            except ValueError:
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

            except Exception as exception:
                print(f"ERROR with request query:    {query}\n{str(exception)}")
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

        return _try_request(query, self.sparql_endpoint)
