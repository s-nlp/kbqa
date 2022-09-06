import time
import sys
import networkx as nx
import matplotlib.pyplot as plt
from caches.base import CacheBase
from SPARQLWrapper import SPARQLWrapper, JSON
from caches.wikidata_entity_to_label import WikidataEntityToLabel
from caches.wikidata_shortest_path import WikidataShortestPathCache

# from mGENRE_MEL.genre.trie import Trie, MarisaTrie
# from mGENRE_MEL.fairseq_model import mGENRE


class SubgraphsRetriever(CacheBase):
    """class for extracting subgraphs given the entities and candidate"""

    def __init__(
        self,
        entity2label: WikidataEntityToLabel,
        shortest_path: WikidataShortestPathCache,
        edge_between_path: bool = False,
        lang: str = "en",
        url: str = "https://query.wikidata.org/sparql",
        cache_dir_path: str = "./cache_store",
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths_edges.pkl")
        self.cache = {}
        self.load_from_cache()
        self.entity2label = entity2label
        self.url = url
        self.shortest_path = shortest_path
        self.edge_between_path = edge_between_path
        self.lang = lang

    def get_path(self, entity, candidate):
        """
        return the shortest path from the given entity to the candidate
        """
        path = self.shortest_path.get_shortest_path(entity, candidate)
        path_clean = []

        # extracting the entity ID only
        for node in path:
            entity_id = node.split("/")[-1]
            print(entity_id)
            # in case we see nodes like P136-blah-blah
            entity_id = entity_id.split("-")[0]
            path_clean.append(entity_id)
        return path_clean

    def get_paths(self, e_1, e_2):
        """
        return all shortest paths from the given entity to the candidate
        """
        paths = []

        if isinstance(e_1, list):  # entity2candidate
            entities = e_1
            candidate = e_2
            for entity in entities:
                paths.append(self.get_path(entity, candidate))
        else:  # candidate2entity
            candidate = e_1
            entities = e_2
            for entity in entities:
                paths.append(self.get_path(entity=candidate, candidate=entity))

        return paths

    def get_undirected_shortest_path(self, entity2candidate, candidate2entity):
        """
        return the shortest paths in both direction
        """
        res = []
        for e2c, c2e in zip(entity2candidate, candidate2entity):
            if not e2c and not c2e:
                raise Exception("NO SHORTEST PATH FOUND")

            if e2c and c2e:
                # see which path is shorter
                shorter_path = e2c if len(e2c) < len(c2e) else c2e
            else:
                # shorter path is the non empty list
                shorter_path = e2c if not c2e else c2e
            res.append(shorter_path)
        return res

    def get_subgraph(self, entities, candidate):
        """
        extract subgraphs given all shortest paths and candidate
        """
        entity2candidate = self.get_paths(entities, candidate)
        candidate2entity = self.get_paths(candidate, entities)

        paths = self.get_undirected_shortest_path(entity2candidate, candidate2entity)

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
        h_vertices = set()
        for path in paths:
            for entity in path:
                h_vertices.add(entity)

        res = self.fill_edges_in_subgraph(h_vertices)

        return res

    def fill_edges_in_subgraph(self, vertices):
        """
        given the set of nodes, fill the edges between all the nodes
        """
        res = nx.DiGraph()
        for entity in vertices:
            res.add_node(entity)

        for entity in vertices:
            edges = self.get_edges(entity)

            for result in edges["results"]["bindings"]:
                neighbor_entity = result["o"]["value"].split("/")[-1]
                curr_edge = result["p"]["value"].split("/")[-1]

                if neighbor_entity in vertices:
                    res.add_edge(entity, neighbor_entity, label=curr_edge)

        return res

    def subgraph_without_connection(self, paths):
        """
        combine the shortest paths without the connection between the paths
        """
        res = nx.DiGraph()
        for path in paths:
            # shortest path doesn't return the edges -> fetch the edge for the
            # current short paths
            curr_path = self.fill_edges_in_subgraph(path)

            # combine the currnet subgraph to res subgraph
            res = nx.compose(res, curr_path)
        return res

    def get_edges(self, entity):
        """
        fetch all of the edges for the current vertice
        """
        # entity already in cache, fetch it
        if entity in self.cache:
            return self.cache.get(entity)

        try:
            user_agent = "WDQS-example Python/%s.%s" % (
                sys.version_info[0],
                sys.version_info[1],
            )
            # query to get properties of the current entity
            query = """
            SELECT ?p ?o ?label WHERE 
            {
                BIND(wd:VALUE AS ?q)
                ?q ?p ?o .
                ?o rdfs:label ?label .
                filter( LANG(?label) = 'LANGUAGE' )
            }
            GROUP BY ?p ?o ?label
            """
            query = query.replace("VALUE", entity)
            query = query.replace("LANGUAGE", self.lang)

            sparql = SPARQLWrapper(self.url, agent=user_agent)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            edges = sparql.query().convert()

            # saving the edges in the cache
            self.cache[entity] = edges
            self.save_cache()
            return edges

        except ValueError:
            print("sleep 60...")
            time.sleep(60)
            return self.get_edges(query)

        except Exception:
            print(f"ERROR with request query:    {query}")
            print("sleep 60...")
            time.sleep(60)
            return self.get_edges(query)

    def visualize_subgraph(self, graph, entities, candidate):
        """
        plot the subgraph
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos)
        labels = nx.get_edge_attributes(graph, "label")

        # red for candidate, green for entities, pink for everything else
        color_map = []
        for node in graph:
            if node == candidate:
                color_map.append("red")
            elif node in entities:
                color_map.append("green")
            else:
                color_map.append("pink")

        nx.draw(
            graph,
            pos,
            edge_color="black",
            arrowsize=10,
            node_size=300,
            node_color=color_map,
            alpha=0.9,
            labels={node: self.entity2label.get_label(node) for node in graph.nodes()},
        )

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={
                edge: self.entity2label.get_label(weight)
                for edge, weight in labels.items()
            },
            font_color="red",
        )
        plt.axis("off")

        return plt
