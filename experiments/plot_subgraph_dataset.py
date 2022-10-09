from matplotlib import pyplot as plt
import pickle
import networkx as nx


with open("/workspace/kbqa/subgraphs_dataset/subgraphs_edges_between.pkl", "rb") as f:
    data = pickle.load(f)


def visualize_subgraph(graph):
    """
    plot the subgraph
    """
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)

    plt.axis("off")

    return plt


for i, item in enumerate(data):
    graph = visualize_subgraph(item)
    plt.savefig("graph{}.png".format(i), format="PNG")
