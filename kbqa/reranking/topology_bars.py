# pylint: disable=no-member

import json
import networkx as nx
import pathlib
from tqdm import tqdm
import pandas as pd
from kbqa.wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from gtda.homology import FlagserPersistence
from gtda.graphs import GraphGeodesicDistance
import numpy as np
import warnings
import matplotlib.pyplot as plt
import gudhi
from statistics import variance, mean
from scipy.stats import entropy
import argparse
import re
from kbqa.reranking.baseline import train_classifiers
import seaborn as sns

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()


parser.add_argument(
    "--json_path",
    default="/workspace/kbqa/subgraphs_dataset/dataset_v0_json/subgraph_segments/",
    type=str,
)

parser.add_argument(
    "--meta",
    default="/workspace/kbqa/metas.csv",
    type=str,
)

parser.add_argument(
    "--base_dataset",
    default="/workspace/kbqa/subgraph_data.csv",
    type=str,
)

parser.add_argument(
    # use tda for tda only features
    "--features",
    default="nx_tda",
    type=str,
)

parser.add_argument(
    "--plot_stats",
    default=False,
    type=bool,
)

## Fetching the Subgraphs from Dataset
def get_subgraph_from_json(json_data):

    """
    function to read json files for getting data and constructing Digraph
    function should output a dictionary with subgraph, node_labels, edge_ids, edge_labels, node_type, node_ids
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

    # make dictonary to output subgraph, node_labels, edge_ids, edge_labels, node_type, node_ids
    graph_dict = {
        "subgraph": subgraph,
        "node_labels": node_labels,
        "edge_ids": edge_ids,
        "edge_labels": edge_labels,
        "node_type": node_type,
        "node_ids": node_ids,
    }

    return graph_dict


# getting homology bars from graph
def persistence_bars(graph_dict):
    """
    function to calculate persistence bars using directed nature of the graph
    graph_dict: dictionary with subgraph, node_labels, edge_ids, edge_labels, node_type, node_ids
    subgraph: networkx directed graph
    node_labels: dictionary with node id as key and node label as value
    edge_ids: list of tuples
    edge_labels: list of edge labels
    node_type: list of node types
    node_ids: list of node ids
    returns: np array of persistence bars
    """
    adj_mat = nx.adjacency_matrix(graph_dict["subgraph"])
    geo_dist = GraphGeodesicDistance(directed=True, unweighted=True).fit_transform(
        [adj_mat]
    )
    flagger_pers = FlagserPersistence().fit_transform(geo_dist)
    rips_complex = gudhi.RipsComplex(points=flagger_pers[0])
    simplex_tree = rips_complex.create_simplex_tree()
    persistence_bars = simplex_tree.persistence()
    return persistence_bars


# plotting bars
def display_persistence_bars(persistence_bars):
    """
    function to display persistence bars
    persistence_bars: np array of persistence bars
    returns: plot of persistence bars
    """
    _, axis = plt.subplots(figsize=(10, 10))
    plot = gudhi.plot_persistence_barcode(persistence_bars, ax=axis)
    return plot


# plot statistics
def plot_statistics(dataframe, columns):
    """
    function to plot statistics for features extracted from subgraphs
    """
    new_df = dataframe[columns]

    # calculate some statistics from the new dataframe
    mean_values = new_df.mean()
    median_values = new_df.median()
    std_values = new_df.std()

    # visualize the mean, median, and standard deviation values using a bar plot
    fig1, ax1 = plt.subplots()
    mean_values.plot(kind="bar", color="blue", alpha=0.5, label="Mean", ax=ax1)
    median_values.plot(kind="bar", color="green", alpha=0.5, label="Median", ax=ax1)
    std_values.plot(
        kind="bar", color="red", alpha=0.5, label="Standard Deviation", ax=ax1
    )
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Value")
    ax1.set_title("statistics of TDA features")
    ax1.legend()

    # visualize the correlations between the selected columns using a heatmap
    fig2, ax2 = plt.subplots()
    corr_matrix = new_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("correlation of TDA features")

    return fig1, fig2


if __name__ == "__main__":

    args = parser.parse_args()

    entity2label = WikidataEntityToLabel(
        sparql_endpoint="http://localhost:7200/repositories/wikidata"
    )

    JSON_PATH = args.json_path
    META_PATH = args.meta

    ## Parsing Subgraph Datasets and Fetch Bar Graphs Objects
    graphs_path = pathlib.Path(JSON_PATH).glob("graph_id_*")

    features = {}
    for graph_path in tqdm(list(graphs_path)):

        with open(graph_path, "r") as f:

            graph_data = json.load(f)

            graph = get_subgraph_from_json(graph_data)

            persistence_bar = persistence_bars(graph)

            index = int(re.findall(r"\d+", str(graph_path))[-1])

            number_of_homology_groups = len(persistence_bar)

            bar_lengths = []
            for _, bar in persistence_bar:
                CURR_MAX = bar[1] - bar[0]
                if CURR_MAX == np.inf:
                    CURR_MAX = 10
                bar_lengths.append(CURR_MAX)
            num_features = [
                number_of_homology_groups,
                sum(bar_lengths),
                mean(bar_lengths),
                variance(bar_lengths),
                entropy(bar_lengths, 2),
            ]

            features[index] = num_features

    df = pd.DataFrame(features).T
    df = df.rename(
        columns={
            0: "number_of_homology_groups",
            1: "sum_of_bar_lengths",
            2: "mean_bar_lengths",
            3: "var_bar_lengths",
            4: "entropy_of_bar_lengths_for_nodes",
        }
    )

    base_data = pd.read_csv(args.base_dataset)
    metas = pd.read_csv(META_PATH)

    data = pd.merge(metas, base_data, left_index=True, right_index=True)
    data = data.sort_values("idx")
    data["label"] = data["candidate_id"] == data["target_id"]
    data["label"] = data["label"].astype(int)

    data = pd.merge(data, df, left_index=True, right_index=True)
    data = data.sort_values("idx")

    data.to_csv("merged_data.csv")

    numeric_features = data[
        [
            "number_of_triangles",
            "number_of_nodes",
            "number_of_edges",
            "candidate_katz_centrality",
            "candidate_pagerank",
            "subgraph_largest_clique_size",
            "shortest_path_lengths",
            "mean_shortest_path_length",
            "sitelinks",
            "outcoming_links",
            "incoming_links",
            "candidate_eigenvector_centrality",
            "candidate_clustering",
            "label",
            "number_of_homology_groups",
            "sum_of_bar_lengths",
            "mean_bar_lengths",
            "var_bar_lengths",
            "entropy_of_bar_lengths_for_nodes",
        ]
    ]
    numeric_features = numeric_features.dropna()

    y_label = numeric_features["label"]
    X_data = numeric_features.drop("label", axis=1)

    clfs_dict = {
        "KNeighborsClassifier": {"n_neighbors": range(3, 17, 2)},
        "RandomForestClassifier": {
            "n_estimators": range(1, 15, 1),
            "max_features": range(1, 16, 1),
        },
        "LogisticRegression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l2"],
            "max_iter": [5000],
        },
        "XGBClassifier": {
            "max_depth": range(2, 10, 1),
            "alpha": range(1, 10, 1),
            "n_estimators": range(60, 220, 40),
            "learning_rate": [0.1, 0.01, 0.05],
        },
    }

    if args.features == "nx_tda":
        for key, value in clfs_dict.items():
            parse_dict = {key: value}
            _, _, f1, b_score, _ = train_classifiers(parse_dict, X_data, y_label)
            print(
                "F1 score for {} = {}, Balanced_Accuracy = {}".format(key, f1, b_score)
            )

    # only using TDA features
    TDA_features = [
        "label",
        "number_of_homology_groups",
        "sum_of_bar_lengths",
        "mean_bar_lengths",
        "var_bar_lengths",
        "entropy_of_bar_lengths_for_nodes",
    ]

    numeric_features = data[TDA_features]
    numeric_features = numeric_features.dropna()

    y_label = numeric_features["label"]
    X_data = numeric_features.drop("label", axis=1)

    if args.features == "tda":
        for key, value in clfs_dict.items():
            parse_dict = {key: value}
            _, _, f1, b_score, _ = train_classifiers(parse_dict, X_data, y_label)
            print(
                "F1 score for only TDA {} = {}, Balanced_Accuracy = {}".format(
                    key, f1, b_score
                )
            )

    if args.plot_stats:
        fig1, fig2 = plot_statistics(data, TDA_features)
        fig1.savefig("tda_stats.png")
        fig2.savefig("tda_correlation.png")
