from caches.base import CacheBase
import requests
import time
from requests.exceptions import JSONDecodeError


class WikidataShortestPathCache(CacheBase):
    """WikidataShortestPathCache - class for request shortest path from wikidata service
    with storing cache
    """

    def __init__(self, cache_dir_path: str = "./cache_store") -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths.pkl")
        self.cache = {}
        self.load_from_cache()

    def get_shortest_path(self, item1, item2):
        if item1 is None or item2 is None:
            return None

        key = (item1, item2)

        if key in self.cache:
            return self.cache[key]

        path, _ = self._request_depth(key[0], key[1])
        self.cache[key] = path
        self.save_cache()

        return self.cache[key]

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
            gas:maxIterations 10 ;      
            gas:maxVisited 10000 .                            
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

            except Exception:
                print(f"ERROR with request query:    {query}")
                print("sleep 60...")
                time.sleep(60)
                return _try_get_depth(query, url)

        return _try_get_depth(query, url)
