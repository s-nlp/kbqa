import time
import sys
import networkx as nx
from caches.wikidata_entity_to_label import WikidataEntityToLabel
from caches.genre import GENREWikidataEntityesCache
from caches.wikidata_shortest_path import WikidataShortestPathCache
from SPARQLWrapper import SPARQLWrapper, JSON
from caches.base import CacheBase

# from mGENRE_MEL.genre.trie import Trie, MarisaTrie
# from mGENRE_MEL.fairseq_model import mGENRE


class SubgraphsRetriever(CacheBase):
    """class for extracting subgraphs given the entities and candidate"""

    def __init__(
        self, Entity2label, Shortest_path, cache_dir_path: str = "./cache_store"
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths_edges.pkl")
        self.cache = {}
        self.load_from_cache()
        self.entity2label = Entity2label
        self.shortest_path = Shortest_path

    def get_paths(self, shortest_path, entities, candidate):
        """
        return the shortest paths from the given entity to the candidate
        """
        paths = []
        for entity in entities:
            paths.append(shortest_path.get_shortest_path(entity, candidate))
        return paths

    def get_subgraphs(self, entities, candidate):
        """
        extract subgraphs given all shortest paths and candidate
        """
        paths = self.get_paths(self.shortest_path, entities, candidate)

        # distinct set of our entities in the paths
        h_vertices = set()
        for path in paths:
            for entity in path:
                h_vertices.add(entity.split("/")[-1])

        res = {}
        for entity in h_vertices:
            res[entity] = set()

        url = "https://query.wikidata.org/sparql"

        for entity in h_vertices:
            print(
                "dealing with entity: {}, val: {}".format(
                    entity, self.entity2label.get_label(entity)
                )
            )

            # load from cache if possible
            if entity not in self.cache:
                edges = self.get_edges(entity, url)
                self.cache[entity] = edges
                self.save_cache()
            else:
                edges = self.cache[entity]

            for result in edges["results"]["bindings"]:
                neighbor_entity = result["o"]["value"].split("/")[-1]
                curr_edge = result["p"]["value"]

                print(
                    "neighbor node wd: {} - {} with edge: {}".format(
                        neighbor_entity,
                        self.entity2label.get_label(neighbor_entity),
                        curr_edge,
                    )
                )
                if neighbor_entity in h_vertices:
                    res[entity].add(neighbor_entity)
                    res[neighbor_entity].add(entity)

        # convert to graphx
        res = nx.MultiDiGraph(res)
        self.cache["subgraph"] = res
        self.save_cache()
        return res

    def get_edges(self, entity, url):
        """
        function to feth all of the edges for the current vertice
        """
        try:
            user_agent = "WDQS-example Python/%s.%s" % (
                sys.version_info[0],
                sys.version_info[1],
            )
            # query to get properties of the current entity
            query = """SELECT ?p ?o ?label WHERE 
{
  BIND(wd:Q22686 AS ?q)
  ?q ?p ?o .
  ?o rdfs:label ?label .
  filter( LANG(?label) = 'en' )
}
GROUP BY ?p ?o ?label"""

            query = query.replace("VALUE", entity)

            # TODO adjust user agent; see https://w.wiki/CX6
            sparql = SPARQLWrapper(url, agent=user_agent)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            return sparql.query().convert()

        except ValueError:
            print("sleep 60...")
            time.sleep(60)
            return self.get_edges(query, url)

        except Exception:
            print(f"ERROR with request query:    {query}")
            print("sleep 60...")
            time.sleep(60)
            return self.get_edges(query, url)


if __name__ == "__main__":
    # sample question: in what french city did antoine de févin die
    E1 = "Q22686"  # Doland Trump
    E2 = "Q242351"  # Ivanka Trump
    E3 = "Q432473"  # Melania Trump
    Es = [E1, E2, E3]

    C = "Q30"  # USA
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()
    SubgraphsRetriever = SubgraphsRetriever(entity2label, shortest_path)
    subgraph = SubgraphsRetriever.get_subgraphs(Es, C)

