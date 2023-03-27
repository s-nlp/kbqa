# pylint: disable = import-error

"""
module for plotting subgraphs
"""
import networkx as nx
import matplotlib.pyplot as plt
from ..wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from ..wikidata.wikidata_subgraphs_retriever import SubgraphNodeType
import graphviz


def nx_subgraph(graph, node_labels, edge_ids, edge_labels, node_types):

    """
    function to plot networkx Digraph with variable colors and node sizes
    """
    plt.figure(figsize=(12, 10))
    pos = nx.shell_layout(graph)

    color_map = []
    node_sizes = []
    for node in node_types:
        if node == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:
            color_map.append("coral")
            node_sizes.append(5000)
        elif node == SubgraphNodeType.QUESTIONS_ENTITY:
            color_map.append("deepskyblue")
            node_sizes.append(2500)
        else:
            color_map.append("lightgray")
            node_sizes.append(800)

    nx.draw_networkx(
        graph,
        pos,
        node_size=node_sizes,
        node_color=color_map,
        edge_color="black",
        arrowsize=10,
        font_weight="bold",
        font_size=8,
        alpha=0.9,
        labels=node_labels,
        arrows=True,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=dict(zip(edge_ids, edge_labels)),
        font_color="red",
        font_weight="bold",
    )
    plt.axis("off")

    return plt


def graphviz_subgraph(node_labels, edge_ids, edge_labels, node_types):

    """
    function to plot colored graphviz graph
    """
    entity2label = WikidataEntityToLabel()

    viz_subgraph = graphviz.Digraph(format="png")

    node_labels_values = [value for key, value in node_labels.items()]
    node_labels_values = ["None" if v is None else v for v in node_labels_values]

    edge_ids_labels = []
    for item in edge_ids:
        edge_1, edge_2 = item
        edge_1_label = entity2label.get_label(edge_1)
        edge_2_label = entity2label.get_label(edge_2)
        edge_ids_labels.append(tuple((edge_1_label, edge_2_label)))

    for node, type_node in zip(node_labels_values, node_types):

        if type_node == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:

            viz_subgraph.node(f'"{node}"', color="coral", style="filled")
        elif type_node == SubgraphNodeType.QUESTIONS_ENTITY:
            viz_subgraph.node(f'"{node}"', color="deepskyblue", style="filled")
        else:
            viz_subgraph.node(f'"{node}"', color="lightgrey", style="filled")

    for edge_id, edge_label in zip(edge_ids_labels, edge_labels):

        source, target = edge_id

        viz_subgraph.edge(
            f'"{source}"',
            f'"{target}"',
            label=f'"{edge_label}"',
        )

    viz_subgraph.edge_attr.update(arrowhead="vee", arrowsize="1")

    return viz_subgraph
