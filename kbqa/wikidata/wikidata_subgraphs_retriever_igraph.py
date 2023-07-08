# pylint: disable=c-extension-no-member
from typing import List, Optional
from .wikidata_entity_to_label import WikidataEntityToLabel
from .wikidata_shortest_path_igraph import WikidataShortestPathIgraphCache
from .wikidata_subgraphs_retriever import SubgraphNodeType
from igraph import Graph, union, plot
import matplotlib.pyplot as plt


class SubgraphsRetrieverIgraph:
    """class for extracting subgraphs given the entities and candidate"""

    def __init__(
        self,
        entity2label: WikidataEntityToLabel,
        shortest_path: WikidataShortestPathIgraphCache,
        edge_between_path: bool = True,
        lang: str = "en",
    ):
        self.cache = {}
        # self.load_from_cache()
        self.entity2label = entity2label
        self.edge_between_path = edge_between_path
        self.lang = lang
        self.shortest_path = shortest_path

    def get_subgraph(
        self, entities: List[str], candidate: str, number_of_pathes: Optional[int] = 10
    ):
        """
        get subgraph given entities and candidates
        """
        paths = []
        for entity in entities:
            path = self.shortest_path.get_shortest_path(
                entity,
                candidate,
            )
            paths.extend(path[:number_of_pathes])

        if self.edge_between_path is True:
            graph = self.subgraph_with_connection(paths)
        else:
            graph = self.subgraph_without_connection(paths)

        # classify the node type
        for vertex in graph.vs:
            if vertex["name"] == candidate:
                vertex["node_type"] = SubgraphNodeType.ANSWER_CANDIDATE_ENTITY
            elif vertex["name"] in entities:
                vertex["node_type"] = SubgraphNodeType.QUESTIONS_ENTITY
            else:
                vertex["node_type"] = SubgraphNodeType.INTERNAL
        return graph, paths

    def subgraph_without_connection(self, paths):
        """subgraphs without connections between shortest paths"""
        graphs = []
        for path in paths:
            graph = self.shortest_path_to_graph(path)
            graphs.append(graph)
        res = union(graphs, byname=True)
        return res

    def subgraph_with_connection(self, paths):
        # first get subgraph without connections in between
        graph = self.subgraph_without_connection(paths)

        for idx, path in enumerate(paths):  # pylint: disable=too-many-nested-blocks
            # other path beside than the current one we're on
            other_paths = paths[:idx] + paths[idx + 1 :]
            other_nodes = sum(other_paths, [])  # flatten 2d list to 1d
            for curr_node in path:
                for other_node in other_nodes:
                    if curr_node != other_node:
                        edge_to = self.shortest_path.get_edge(curr_node, other_node)
                        if edge_to is not None:
                            graph = self.add_edge(graph, edge_to, curr_node, other_node)
                        else:  # check the other way if edge_to is none
                            edge_from = self.shortest_path.get_edge(
                                other_node, curr_node
                            )
                            if edge_from is not None:
                                graph = self.add_edge(
                                    graph, edge_from, other_node, curr_node
                                )
        return graph

    def add_edge(self, graph, edge, node1, node2):
        """
        add the edge between node1 and node2 if it exists
        """
        # we do have an edge between these 2 nodes
        node1 = graph.vs.select(name=node1)
        node2 = graph.vs.select(name=node2)
        node1_name, node2_name = node1["name"][0], node2["name"][0]

        # only add edge if there exist no edges already between these nodes
        if not graph.are_connected(node1_name, node2_name):
            edge_labels = self.entity2label.get_label(edge)
            graph.add_edge(
                node1_name,
                node2_name,
                edge_id=edge,
                edge_label=edge_labels,
            )
        return graph

    def entity_ids_to_label(self, entities):
        """
        list of wikidata entity id to natural language labels
        """
        res = []
        for entity in entities:
            lab = self.entity2label.get_label(entity)
            if lab is None:  # can't find label, just return id
                lab = entity
            res.append(lab)
        return res

    def shortest_path_to_graph(self, path):
        """
        given the shortest path, return the igraph graph equivalence
        """
        graph = Graph(directed=True)
        vertex_labels = self.entity_ids_to_label(path)
        # name is wikidata id, label is the english label
        graph.add_vertices(
            len(path),
            attributes={"name": path, "label": vertex_labels},
        )

        # getting the edges between nodes
        edge_direction = []
        edge_ids = []
        edge_labels = []
        for idx, curr_node in enumerate(path):
            if idx < len(path) - 1:
                # getting edge id
                edge_id = self.shortest_path.get_edge(curr_node, path[idx + 1])

                # only add edge if there exist an edge between nodes
                if edge_id is not None:
                    edge_direction.append((idx, idx + 1))
                    edge_ids.append(edge_id)

                    # getting edge label
                    edge_label = self.entity2label.get_label(edge_id)
                    if edge_label is None:  # can't find label, just return id
                        edge_label = edge_id
                    edge_labels.append(edge_label)

        graph.add_edges(
            edge_direction, attributes={"edge_id": edge_ids, "edge_label": edge_labels}
        )
        return graph

    @staticmethod
    def visualize(graph, output_path):
        my_layout = graph.layout_fruchterman_reingold()
        plot(
            graph,
            vertex_size=20,
            vertex_color="steelblue",
            vertex_label=graph.vs["label"],
            edge_label=graph.es["edge_label"],
            edge_background="white",
            edge_align_label=True,
            target=output_path,
            layout=my_layout,
        )
        plt.show()
