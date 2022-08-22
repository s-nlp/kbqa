import requests
import time
from caches.base import CacheBase


class WikidataEntityToLabel(CacheBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(self, cache_dir_path: str = "./cache_store") -> None:
        super().__init__(cache_dir_path, "wikidata_entity_to_label.pkl")
        self.cache = {}
        self.load_from_cache()

    def get_label(self, entity_idx):
        if entity_idx not in self.cache:
            label = self._request_wikidata(entity_idx)
            if label is not None:
                self.cache[entity_idx] = label

        return self.cache.get(entity_idx)

    def _request_wikidata(self, entity_idx):
        url = "https://query.wikidata.org/sparql"
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
                request = requests.get(url, params={"format": "json", "query": query})
                data = request.json()
                return data["results"]["bindings"][0]["label"]["value"]

            except ValueError:
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

            except Exception:
                print(f"ERROR with request query:    {query}")
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

        return _try_request(query, url)
