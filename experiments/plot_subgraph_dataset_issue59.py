import pathlib
import pickle
import networkx as nx
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from wikidata.wikidata_shortest_path import WikidataShortestPathCache
from wikidata.wikidata_subgraphs_retriever import SubgraphsRetriever
import matplotlib.pyplot as plt
import os


def fetch_subgraphs_from_pkl(file_path):
    """
    retrieving our subgraphs from pkl file
    """
    with open(file_path, "rb") as file:
        subgraphs = pickle.load(file)
    return subgraphs


if __name__ == "__main__":

    curr_dir = pathlib.Path().resolve()

    # path to subgraph cache
    PATH = str(curr_dir) + "/subgraphs_dataset/subgraphs_edges_between.pkl"

    subgraphs = fetch_subgraphs_from_pkl(PATH)

    entity2label = WikidataEntityToLabel()

    for i, subgraph in enumerate(subgraphs):

        node_type = nx.get_node_attributes(subgraph, "node_type")
        entities = []
        candidate = []
        for node in subgraph:
            if node_type[node] == "entity":
                entities.append(node)
            elif node_type[node] == "candidate":
                candidate.append(node)

        entity2label = WikidataEntityToLabel()
        shortest_path = WikidataShortestPathCache()
        subgraph_obj = SubgraphsRetriever(
            entity2label, shortest_path, edge_between_path=True
        )
        ax = subgraph_obj.visualize_subgraph(subgraph, entities, candidate)

        # creating dataset considering 1 correct and 5 wrong candidates for each question
        SUBGRAPH_PATH = "subgraph_plots"
        if not os.path.exists(SUBGRAPH_PATH):
            os.makedirs(SUBGRAPH_PATH)
        if i == 0 or (i % 5 == 1 and i != 1):
            CORRECT_PATH = os.path.join(SUBGRAPH_PATH, "correct")
            if not os.path.exists(CORRECT_PATH):
                os.makedirs(CORRECT_PATH)
            plt.savefig(CORRECT_PATH + "/{}.svg".format(i))
        else:
            WRONG_PATH = os.path.join(SUBGRAPH_PATH, "wrong")
            if not os.path.exists(WRONG_PATH):
                os.makedirs(WRONG_PATH)
            plt.savefig(WRONG_PATH + "/{}.svg".format(i))

    print("subgraphs were plotted and stored")
