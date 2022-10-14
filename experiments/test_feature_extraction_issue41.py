# pylint: disable=import-error

from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from wikidata.wikidata_shortest_path import WikidataShortestPathCache
from reranking import feature_extraction
import networkx as nx

# Sample Wrong Entities and Candidate
E1 = "Q22686"  # Trump
E2 = "Q36159"  # DC

Es = [E1, E2]

C = "Q4917"  # Dollar

entity2label = WikidataEntityToLabel()
shortest_path = WikidataShortestPathCache()
subgraph_obj = SubgraphsRetriever(entity2label, shortest_path, edge_between_path=True)
subgraph = subgraph_obj.get_subgraph(Es, C)

adj_mat = nx.adjacency_matrix(subgraph, nodelist=None, dtype=None, weight="weight")
g = adj_mat.toarray()
G = nx.from_numpy_array(g)

print(feature_extraction.number_of_triangles(subgraph, True))
print(feature_extraction.nodes_and_edges(subgraph))
print(feature_extraction.katz_centrality(G))
print(feature_extraction.eigenvector_centrality(G))
print(feature_extraction.clustering(G))
print(feature_extraction.k_components(G))
print(feature_extraction.large_clique_size(G))
print(feature_extraction.diameter(G))
print(feature_extraction.communicability(G))
print(feature_extraction.pagerank(G))
