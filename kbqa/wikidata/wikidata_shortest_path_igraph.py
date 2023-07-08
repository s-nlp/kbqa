# pylint: disable=c-extension-no-member
import time
import igraph

from ..logger import get_logger
from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase

logger = get_logger()


class WikidataShortestPathIgraphCache(WikidataBase):
    """WikidataShortestPathCache - class for request shortest path from wikidata service
    with storing cache
    """

    def __init__(
        self,
        graph_path: str,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
    ):
        super().__init__(cache_dir_path, "wikidata_shortest_paths_igraph.pkl")
        start_time = time.time()
        logger.info("loading igraph")
        self.subgraph = igraph.Graph.Read_Ncol(
            graph_path, names=True, directed=True, weights=True
        )
        logger.info(f"took {time.time() - start_time} to load igraph.")
        self.cache = {}
        self.load_from_cache()

    def clean_path(self, paths):
        """
        give the shortest path in igraph id, convert to wikidata id
        """
        res = []
        for path in paths:
            clean_path = []
            for node in path:
                # find the node i igraph
                curr_node_igraph = self.subgraph.vs.find(node)
                curr_node_name = curr_node_igraph["name"]
                clean_path.append(f"Q{str(curr_node_name)}")
            res.append(clean_path)
        return res

    def get_edge(self, entity1: str, entity2: str):
        """
        return the edges between entity1 and entity2
        entity1 and entity2 are wikidata id
        """
        try:
            entity1_id = self.subgraph.vs.find(entity1[1:])  # not including Q
            entity2_id = self.subgraph.vs.find(entity2[1:])  # not including Q

            edge_igraph = self.subgraph.get_eid(entity1_id, entity2_id)
            edge = self.subgraph.es.find(edge_igraph)
            edge = f"P{str(int(edge['weight']))}"
        except igraph._igraph.InternalError:  # pylint: disable=W0212
            logger.error(f"no edge between {entity1}->{entity2}")
            return None
        except ValueError:  # either edge does not exist
            logger.error(
                f"Problem finding edge {entity1}->{entity2}, either one entity does not exist in igraph"
            )
            return None
        return edge

    def get_shortest_path(
        self,
        item1,
        item2,
    ):
        """
        returns the shortest path both ways for item 1 and 2
        """
        item1_id = item1[1:]  # not including Q
        item2_id = item2[1:]  # not including Q

        try:
            key = (item1, item2)
            if key in self.cache:
                paths = self.cache[key]
            else:
                paths = self.subgraph.get_all_shortest_paths(
                    item1_id, to=item2_id, mode="all"
                )
                self.cache[key] = paths
                self.save_cache()
            paths = self.clean_path(paths)
        except ValueError:
            logger.error(
                f"Problem finding path {item1}->{item2}, either one entity does not exist in igraph"
            )
            paths = [[item1, item2]]
        return paths
