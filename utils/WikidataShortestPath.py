import os
import pickle
import time

import requests
from requests.exceptions import JSONDecodeError


class WikidataShortestPath:
    """WikidataShortestPath - class for request shortest path from wikidata service
    with storing cache
    """

    MAX_DEPTH = 1000000000

    def __init__(self, cache_path: str = "./cache") -> None:
        self.shortest_path_len_cache = dict()
        self.paths_cache = dict()

        self.cache_path = cache_path
        self.shortest_path_cache_file_path = os.path.join(
            self.cache_path, "wsp_shortest_path_cache.pkl"
        )
        self.paths_cache_file_path = os.path.join(
            os.path.join(self.cache_path, "wsp_paths_cache.pkl")
        )

        self._load_from_cache()

    def _load_from_cache(self):
        if os.path.exists(self.shortest_path_cache_file_path):
            with open(self.shortest_path_cache_file_path, "rb") as file:
                self.shortest_path_len_cache = pickle.load(file)

        if os.path.exists(self.paths_cache_file_path):
            with open(self.paths_cache_file_path, "rb") as file:
                self.paths_cache = pickle.load(file)

    def _save_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        with open(self.shortest_path_cache_file_path, "wb") as file:
            pickle.dump(self.shortest_path_len_cache, file)

        with open(self.paths_cache_file_path, "wb") as file:
            pickle.dump(self.paths_cache, file)

    def get_shortest_path(self, item1, item2):
        if item1 is None or item2 is None:
            return [], self.MAX_DEPTH

        key = [item1, item2]
        key = sorted(key)
        key = tuple(key)

        if key in self.shortest_path_len_cache and key in self.paths_cache:
            return self.paths_cache[key], self.shortest_path_len_cache[key]

        path, max_depth = self._request_depth(key[0], key[1])
        self.shortest_path_len_cache[key] = max_depth
        self.paths_cache[key] = path
        self._save_cache()

        return self.paths_cache[key], self.shortest_path_len_cache[key]

    def _request_depth(self, item1, item2):
        url = "https://query.wikidata.org/sparql"
        query = """
        select * {
            SERVICE gas:service {     
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ; 
            gas:in wd:%s ;     
            gas:target wd:%s ;        
            gas:out ?out ;      
            gas:out1 ?depth ;     
            gas:maxIterations 4 ;      
            gas:maxVisited 2000 .                            
            }
        }
        """ % (
            item1,
            item2,
        )

        def _try_get_depth(query, url):
            try:
                request = requests.get(url, params={"format": "json", "query": query})
                data = request.json()

                max_depth = 0
                for rec in data["results"]["bindings"]:
                    depth = float(rec["depth"]["value"])
                    max_depth = max(max_depth, depth)

                path = [
                    r["out"]["value"]
                    for r in sorted(
                        data["results"]["bindings"],
                        key=lambda r: float(r["depth"]["value"]),
                    )
                ]

                return path, max_depth

            except JSONDecodeError:
                print("sleep 60...")
                time.sleep(60)
                return _try_get_depth(query, url)

        return _try_get_depth(query, url)
