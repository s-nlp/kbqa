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
    """
    class for extracting subgraphs given the entities and candidate
    edge_between_path=false ->
        subgraphs combined with NO edges from
        unconnected nodes from individual chain
    edge_between_path=True ->
        subgraphs combined WITH edges from
        unconnected nodes from individual chain
    """

    def __init__(
        self,
        Entity2label,
        Shortest_path,
        edge_between_path=False,
        cache_dir_path: str = "./cache_store",
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths_edges.pkl")
        self.cache = {}
        self.load_from_cache()
        self.entity2label = Entity2label
        self.shortest_path = Shortest_path
        self.edge_between_path = edge_between_path

    def get_paths(self, shortest_path, entities, candidate):
        """
        return the shortest paths from the given entity to the candidate
        """
        paths = []
        for entity in entities:
            paths.append(shortest_path.get_shortest_path(entity, candidate))
        return paths

    def get_distinct_entities(self, paths):
        """
        given the shortest paths, return a list of the distinct entities
        """
        res = set()
        for path in paths:
            for entity in path:
                res.add(entity.split("/")[-1])
        return res

    def get_subgraphs(self, entities, candidate):
        """
        extract subgraphs given all shortest paths and candidate
        """
        paths = self.get_paths(self.shortest_path, entities, candidate)
        print(paths)
        if self.edge_between_path is True:
            res = self.subgraph_with_connection(paths)
        else:
            res = self.subgraph_without_connection(paths)

        return res

    def subgraph_with_connection(self, paths):
        """
        combine the shortest paths with the connection between the paths
        """
        # distinct set of our entities in the paths
        h_vertices = self.get_distinct_entities(paths)

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
                self.save_cache()
            else:
                edges = self.cache[entity]

            for result in edges["results"]["bindings"]:
                neighbor_entity = result["p"]["value"].split("/")[-1]
                curr_edge = result["attName"]["value"]

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
        res = nx.DiGraph(res)
        print(res)
        return res

    def subgraph_without_connection(self, paths):
        """
        combine the shortest paths without the connection between the paths
        """
        res = nx.Graph()
        for path in paths:
            shortest_path = nx.Graph()
            for idx, vertice in enumerate(path):
                shortest_path.add_node(vertice)
                # add edge to next v if not at the last nv
                if idx < len(path) - 1:
                    shortest_path.add_edge(vertice, path[idx + 1])
            # combine the currnet subgraph to res subgraph
            res = nx.compose(res, shortest_path)
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
            query = """SELECT ?p ?attName  WHERE {
            BIND(wd:VALUE AS ?q)
            ?q ?p ?statement.
            ?realAtt wikibase:claim ?p.
            ?realAtt rdfs:label ?attName.
            FILTER(((LANG(?attName)) = "en") || ((LANG(?attName)) = ""))
            }
            GROUP BY ?p ?attName"""

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

    def visualize_subgraph(self, graph):
        """
        graph the subgraph with plt
        """
        nx.draw(graph, with_labels=True)
        plt.draw()
        plt.show()


if __name__ == "__main__":
    # sample question: in what french city did antoine de févin die
    E1 = "Q2856873"  # Antoine de Févin
    E2 = "Q7742"  # Louis XIV of France
    E3 = "Q185075"  # Saint-Germain-en-Laye
    Es = [E1, E2, E3]

    C = "Q6441"  # Montpelier
    entity2label = WikidataEntityToLabel()
    shortest_path = WikidataShortestPathCache()
    SubgraphsRetriever = SubgraphsRetriever(entity2label, shortest_path, edge_between_path=True)
    subgraph = SubgraphsRetriever.get_subgraphs(Es, C)
