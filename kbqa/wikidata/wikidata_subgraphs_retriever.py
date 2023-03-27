# pylint: disable=no-else-continue

import enum
import logging
import requests

from typing import List, Optional
from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase
from .wikidata_entity_to_label import WikidataEntityToLabel
from .wikidata_shortest_path import WikidataShortestPathCache


class SubgraphNodeType(str, enum.Enum):
    """SubgraphNodeType - Enum class with types of subgraphs nodes"""

    INTERNAL = "Internal"
    QUESTIONS_ENTITY = "Question entity"
    ANSWER_CANDIDATE_ENTITY = "Answer candidate entity"


class SubgraphsRetriever(WikidataBase):
    """class for extracting subgraphs given the entities and candidate"""

    def __init__(
        self,
        entity2label: WikidataEntityToLabel,
        shortest_path: WikidataShortestPathCache,
        edge_between_path: bool = False,
        num_request_time: int = 3,
        lang: str = "en",
        sparql_endpoint: str = None,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_shortest_paths_edges.pkl", sparql_endpoint
        )
        self.cache = {}
        self.load_from_cache()
        self.entity2label = entity2label
        self.shortest_path = shortest_path
        self.edge_between_path = edge_between_path
        self.lang = lang
        self.num_request_time = num_request_time

    def get_subgraph(
        self, entities: List[str], candidate: str, number_of_pathes: Optional[int] = 10
    ):
        """Extract subgraphs given all shortest paths and candidate

        Args:
            entities (List[str]): List of question entities identifiest
            candidate (str): Identifier of answer candidate entity
            number_of_pathes (Optional[int], optional): maximum number of shortest pathes that will queried from KG
                for each pair question entity and candidate entiry.
                Defaults to None.

        Returns:
            _type_: _description_
        """
        # Query shortest pathes between entities and candidate
        pathes = []
        for entity in entities:
            e2c_pathes = self.shortest_path.get_shortest_path(
                entity,
                candidate,
                return_edges=False,
                return_only_first=False,
                return_id_only=True,
            )
            c2e_pathes = self.shortest_path.get_shortest_path(
                candidate,
                entity,
                return_edges=False,
                return_only_first=False,
                return_id_only=True,
            )

            # If pathes not exist in one way, just take other
            if e2c_pathes is None and c2e_pathes is not None:
                pathes.extend(c2e_pathes[:number_of_pathes])
                continue
            elif c2e_pathes is None and e2c_pathes is not None:
                pathes.extend(e2c_pathes[:number_of_pathes])
                continue
            elif (
                e2c_pathes is None and e2c_pathes is None
            ):  # If no shortest path for both directions
                pathes.extend([[entity, candidate]])
            else:
                # Take shortest version of pathes
                # If lengths of shortest pathes same for bouth directions, will take pathes from Question to candidate
                if len(e2c_pathes[0]) > len(c2e_pathes[0]):
                    pathes.extend(c2e_pathes[:number_of_pathes])
                else:
                    pathes.extend(e2c_pathes[:number_of_pathes])

        if self.edge_between_path is True:
            graph = self.subgraph_with_connection(pathes)
        else:
            graph = self.subgraph_without_connection(pathes)

        # Fill node attributes information
        for node in graph:
            if node == candidate:
                graph.nodes[node][
                    "node_type"
                ] = SubgraphNodeType.ANSWER_CANDIDATE_ENTITY
            elif node in entities:
                graph.nodes[node]["node_type"] = SubgraphNodeType.QUESTIONS_ENTITY
            else:
                graph.nodes[node]["node_type"] = SubgraphNodeType.INTERNAL

        return graph, pathes

    def get_distinct_nodes(self, paths):
        """
        given the said paths, return a set of distinct nodes
        """
        # distinct set of our entities in the paths
        h_vertices = set()
        for path in paths:
            for entity in path:
                h_vertices.add(entity)
        return h_vertices

    def subgraph_with_connection(self, paths):
        """
        combine the shortest paths with the connection between the paths
        """
        # get subgraph w/ no edges between
        graph_no_connection = self.subgraph_without_connection(paths)
        for idx, path in enumerate(paths):
            # getting the other paths beside the ones we're on
            if idx < len(paths):
                other_paths = paths[:idx] + paths[idx + 1 :]
            else:
                other_paths = paths[:idx]
            for entity in path:
                # distinct set of our entities in the paths
                other_paths_nodes = self.get_distinct_nodes(other_paths)
                other_paths_nodes.add(entity)

                # only fill in edges of current node with other paths' nodes
                if len(other_paths_nodes) > 1:
                    graph_no_connection = self.fill_edges_in_subgraph(
                        other_paths_nodes, graph_no_connection
                    )
        return graph_no_connection

    def fill_edges_in_subgraph(self, vertices, graph=None):
        """
        given the set of nodes, fill the edges between all the nodes
        """
        if graph is None:
            res = nx.DiGraph()
            for entity in vertices:
                res.add_node(entity)
        else:
            res = graph

        for entity in vertices:
            edges = self.get_edges(entity)

            for result in edges["results"]["bindings"]:
                neighbor_entity = result["o"]["value"].split("/")[-1]
                curr_edge = result["p"]["value"].split("/")[-1]

                if neighbor_entity in vertices:
                    if entity != neighbor_entity:
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

    def _request_edges(self, entity):
        try:
            # query to get properties of the current entity
            query = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
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

            request = requests.get(
                self.sparql_endpoint,
                params={"format": "json", "query": query},
                timeout=20,
                headers={"Accept": "application/json"},
            )
            edges = request.json()

            # saving the edges in the cache
            self.cache[entity] = edges
            self.save_cache()
            return edges

        except ValueError:
            logging.info("ValueError")
            return None

    def get_edges(self, entity):
        """
        fetch all of the edges for the current vertice
        """
        # entity already in cache, fetch it
        if entity in self.cache:
            return self.cache.get(entity)

        curr_tries = 0
        edges = self._request_edges(entity)

        # continue to request up to a num_request_time
        while edges is None and curr_tries < self.num_request_time:
            curr_tries += 1
            logging.info("sleep 60...")
            time.sleep(60)
            edges = self._request_edges(entity)

        edges = [] if edges is None else edges
        return edges
