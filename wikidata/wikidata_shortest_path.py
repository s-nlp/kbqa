from typing import List
from wikidata.base import WikidataBase
import requests
import time
from config import SPARQL_ENGINE
import base64
from itertools import groupby


class WikidataShortestPathCache(WikidataBase):
    """WikidataShortestPathCache - class for request shortest path from wikidata service
    with storing cache

    Args:
        cache_dir_path (str, optional): Path to directory with caches. Defaults to "./cache_store".
        sparql_endpoint (_type_, optional): URI for SPARQL endpoint.
            If None, will used SPARQL_ENDPOINT from config. Defaults to None.
        engine (str, optional): Engine of provided SPARQL endpoint. Supported GraphDB and Blazegraph only.
            If None, will used SPARQL_ENGINE from config.
            Defaults to None.

    Raises:
        ValueError: If passed wrong string identifier for engine. Supported only 'grapdb' and 'blazegraph'
    """

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
        sparql_endpoint: str = None,
        engine: str = None,
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths.pkl", sparql_endpoint)

        self.engine = engine
        if self.engine is None:
            self.engine = SPARQL_ENGINE

        self.engine = self.engine.lower()
        if self.engine not in ["blazegraph", "graphdb"]:
            raise ValueError(
                f'only "blazegraph" and "graphdb" engines supported, but passed {engine}'
            )

        self.cache = {}
        self.load_from_cache()

    def get_shortest_path(
        self, item1, item2, return_only_first=True, return_edges=False
    ) -> List:
        """get_shortest_path

        Args:
            item1 (str): Identifier of Entity from which path started
            item2 (str): Identifier of End of path Entity
            return_only_first (bool, optional): Graphdb engine, can return a few pathes.
                If False, it will return only first path, if False, it will return all pathes.
                Works only with graphdb engine.
                Defaults to True.
            return_edges (bool, optional): Graphdb engine can return shortes path with edges.
                If False, it will work like Blazegraph, if True

        Returns:
            list: shortest path or list of shortest pathes
        """
        if item1 is None or item2 is None:
            return None

        if self.engine == "blazegraph" and (
            return_only_first is False or return_edges is True
        ):
            raise ValueError(
                "For Blazegraph engine, return_only_first must be only True and return_edges must be False"
            )

        key = (item1, item2)

        if key in self.cache:
            path_data = self.cache[key]
        else:
            path_data = self._request_path_data(key[0], key[1])
            if path_data is not None:
                self.cache[key] = path_data
                self.save_cache()

        if path_data is None:
            return None

        if self.engine == "blazegraph":
            path = [
                r["out"]["value"]
                for r in sorted(
                    path_data,
                    key=lambda r: float(r["depth"]["value"]),
                )
            ]
            return path

        else:
            # pathIndex - index of path. Results can include a lot of pathes
            # edgeIndex - index of edge in each path.
            # path_data sorted by (pathIndex, edgeIndex)
            pathes = [
                (
                    val["pathIndex"]["value"],
                    val["edgeIndex"]["value"],
                    self._rdf4j_edge_value_decode(val["edge"]["value"]),
                )
                for val in path_data
            ]
            pathes = [list(group) for _, group in groupby(pathes, key=lambda k: k[0])]
            pathes = [[path_el[-1] for path_el in path] for path in pathes]

            if not return_edges:
                pathes = [
                    [path[0][0]] + [path_step[-1] for path_step in path]
                    for path in pathes
                ]

            if return_only_first:
                return pathes[0]
            else:
                return pathes

    def _request_path_data(self, item1, item2):
        if self.engine == "blazegraph":
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
        elif self.engine == "graphdb":
            query = """
            PREFIX path: <http://www.ontotext.com/path#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX wd: <http://www.wikidata.org/entity/>

            SELECT ?pathIndex ?edgeIndex ?edge
            WHERE {
                SERVICE path:search {
                    [] path:findPath path:shortestPath ;
                    path:sourceNode wd:%s ;
                    path:destinationNode wd:%s ;
                    path:pathIndex ?pathIndex ;
                    path:resultBindingIndex ?edgeIndex ;
                    path:resultBinding ?edge ;
                    .
                }
            }
            """ % (
                item1,
                item2,
            )
        else:
            raise ValueError(
                f'only "blazegraph" and "graphdb" engines supported, but passed {self.engine}'
            )

        def _try_get_path_data(query, url):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    headers={"Accept": "application/json"},
                )
                data = request.json()

                if len(data["results"]["bindings"]) == 0:
                    return None
                else:
                    return data["results"]["bindings"]

            except ValueError:
                print("sleep 60...")
                time.sleep(60)
                return _try_get_path_data(query, url)

            except Exception as exception:
                print(f"ERROR with request query:    {query}\n{str(exception)}")
                print("sleep 60...")
                time.sleep(60)
                return _try_get_path_data(query, url)

        return _try_get_path_data(query, self.sparql_endpoint)

    def _rdf4j_edge_value_decode(self, rdf4j_edge):
        edge = base64.urlsafe_b64decode(rdf4j_edge.split(":")[-1])
        edge = edge.decode()[2:-2].split(" ")
        return [val[1:-1] for val in edge]
