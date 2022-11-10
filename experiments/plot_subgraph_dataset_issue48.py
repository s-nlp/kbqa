import json
import networkx as nx
from wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from wikidata.wikidata_subgraphs_retriever import SubgraphNodeType
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import os
from reranking.graph_plotting import nx_subgraph, graphviz_subgraph

parser = argparse.ArgumentParser()

parser.add_argument(
    "--json_graphs_directory",
    default="/workspace/kbqa/subgraphs_dataset/dataset_v0_json/subgraph_segments/",
    type=str,
)

parser.add_argument(
    "--meta_files_directory",
    default="/workspace/kbqa/subgraphs_dataset/dataset_v0_json/meta_segments/",
    type=str,
)

parser.add_argument(
    "--graph_type",
    default="nx",
    type=str,
)


def get_subgraph_from_json(json_data):

    """
    function to read json files for getting data and constructing Digraph
    """

    subgraph = nx.DiGraph()

    node_ids = []
    node_labels = {}
    node_type = []
    for node in json_data["nodes"]:
        node_labels[node["id"]] = entity2label.get_label(node["id"])
        node_ids.append(node["id"])
        node_type.append(node["node_type"])

    subgraph.add_nodes_from(node_ids)

    edge_ids = []
    edge_labels = []
    for edge in json_data["links"]:
        edge_labels.append(entity2label.get_label(edge["label"]))
        edge_ids.append(tuple((edge["source"], edge["target"])))

    subgraph.add_edges_from(edge_ids)

    return subgraph, node_labels, edge_ids, edge_labels, node_type, node_ids


if __name__ == "__main__":

    args = parser.parse_args()

    JSON_PATH = args.json_graphs_directory
    META_PATH = args.meta_files_directory

    metas_path = pathlib.Path(META_PATH).glob("meta_id_*")
    metas = []
    for meta_path in metas_path:
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            metas.append(meta_data)

    metas_df = pd.DataFrame(metas)  # constructing metas dataframe

    curr_dir = pathlib.Path().resolve()

    SUBGRAPH_PATH = "subgraph_plots"
    if not os.path.exists(SUBGRAPH_PATH):
        os.makedirs(SUBGRAPH_PATH)

    graphs_path = pathlib.Path(JSON_PATH).glob("graph_id_*")

    entity2label = WikidataEntityToLabel(
        sparql_endpoint="http://localhost:7200/repositories/wikidata"
    )

    for graph_path in tqdm(graphs_path):
        with open(graph_path, "r") as f:

            graph_data = json.load(f)

            (
                graph,
                node_labels,
                edge_ids,
                edge_labels,
                node_types,
                node_ids,
            ) = get_subgraph_from_json(graph_data)

            # Creating directories according to graph_type flag for storing plot images
            if args.graph_type == "nx":

                NX_PATH = os.path.join(SUBGRAPH_PATH, "nx")
                if not os.path.exists(NX_PATH):
                    os.makedirs(NX_PATH)

                NX_PATH_CORRECT = os.path.join(NX_PATH, "correct")
                if not os.path.exists(NX_PATH_CORRECT):
                    os.makedirs(NX_PATH_CORRECT)

                NX_PATH_WRONG = os.path.join(NX_PATH, "wrong")
                if not os.path.exists(NX_PATH_WRONG):
                    os.makedirs(NX_PATH_WRONG)

                nx_plot = nx_subgraph(
                    graph, node_labels, edge_ids, edge_labels, node_types
                )

                for node_type, node_id in zip(node_types, node_ids):
                    if node_type == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:
                        if node_id in metas_df["target_id"].tolist():
                            nx_plot.savefig(
                                NX_PATH_CORRECT
                                + "/target candidate {}.png".format(node_id)
                            )
                        else:
                            nx_plot.savefig(
                                NX_PATH_WRONG
                                + "/target candidate {}.png".format(node_id)
                            )

            if args.graph_type == "viz":

                GRAPHVIZ_PATH = os.path.join(SUBGRAPH_PATH, "graphviz")
                if not os.path.exists(GRAPHVIZ_PATH):
                    os.makedirs(GRAPHVIZ_PATH)

                GRAPHVIZ_PATH_CORRECT = os.path.join(GRAPHVIZ_PATH, "correct")
                if not os.path.exists(GRAPHVIZ_PATH_CORRECT):
                    os.makedirs(GRAPHVIZ_PATH_CORRECT)

                GRAPHVIZ_PATH_WRONG = os.path.join(GRAPHVIZ_PATH, "wrong")
                if not os.path.exists(GRAPHVIZ_PATH_WRONG):
                    os.makedirs(GRAPHVIZ_PATH_WRONG)

                viz_plot = graphviz_subgraph(
                    node_labels, edge_ids, edge_labels, node_types
                )

                # saving the graphs into appropriate directories created above
                for node_type, node_id in zip(node_types, node_ids):
                    if node_type == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:
                        if node_id in metas_df["target_id"].tolist():
                            viz_plot.render(
                                GRAPHVIZ_PATH_CORRECT
                                + "/target candidate {}".format(node_id),
                                format="png",
                            )
                        else:
                            viz_plot.render(
                                GRAPHVIZ_PATH_WRONG
                                + "/target candidate {}".format(node_id),
                                format="png",
                            )

    print("subgraphs were plotted and stored")
