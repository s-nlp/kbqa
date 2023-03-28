# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from kbqa.reranking.feature_extraction import *
import pathlib
import json
import networkx as nx
import csv
from tqdm import tqdm
import argparse
import pandas as pd
from kbqa.wikidata.wikidata_entity_to_label import WikidataEntityToLabel
from kbqa.wikidata.wikidata_subgraphs_retriever import SubgraphNodeType
import re
from kbqa.reranking.get_sitelinks import *

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


def fetch_subgraphs_from_json(file_path):
    """
    retrieving our subgraphs from json file
    """
    with open(file_path, "r") as file:
        subgraphs = json.load(file)
    return subgraphs


def fetch_metas_from_json(file_path):
    """
    retrieving our subgraphs from json file
    """
    with open(file_path, "r") as file:
        meta = json.load(file)
    return meta


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

    entity2label = WikidataEntityToLabel(
        sparql_endpoint="http://localhost:7200/repositories/wikidata"
    )

    metas_path = pathlib.Path(META_PATH).glob("meta_id_*")
    metas = []

    for meta_path in metas_path:
        metas.append(fetch_metas_from_json(meta_path))

    metas_df = pd.DataFrame(metas)

    graphs_path = pathlib.Path(JSON_PATH).glob("graph_id_*")
    list_graphs_path = list(graphs_path)

    data = []
    candidates = []

    for graph_path in tqdm(list_graphs_path):

        index = int(re.findall(r"\d+", str(graph_path))[-1])

        json_graph = fetch_subgraphs_from_json(graph_path)

        subgraph, _, _, _, node_types, node_ids = get_subgraph_from_json(json_graph)

        adj_mat = nx.adjacency_matrix(
            subgraph, nodelist=None, dtype=None, weight="weight"
        )

        g = adj_mat.toarray()

        G = nx.from_numpy_array(g)

        N_TRIANGLES = number_of_triangles(subgraph, True)

        n_nodes, n_edges = nodes_and_edges(subgraph)

        for node_type, node_id in zip(node_types, node_ids):
            if node_type == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:
                candidates.append(node_id)
        if not candidates:
            candidates.append(None)

        f_katz_centrality = katz_centrality(subgraph)
        N_KATZ_CENTRALITY = 0

        for key, value in f_katz_centrality.items():
            if key == candidates[-1]:
                N_KATZ_CENTRALITY += value

        f_eigenvector_centrality = eigenvector_centrality(subgraph)
        N_EIGENVECTOR_CENTRALITY = 0

        for key, value in f_eigenvector_centrality.items():
            if key == candidates[-1]:
                N_EIGENVECTOR_CENTRALITY += value

        f_clustering = clustering(subgraph)
        N_CLUSTERING = 0

        for key, value in f_clustering.items():
            if key == candidates[-1]:
                N_CLUSTERING += value

        f_pagerank = pagerank(subgraph)
        N_PAGERANK = 0

        for key, value in f_pagerank.items():
            if key == candidates[-1]:
                N_PAGERANK += value

        n_largest_clique_size = large_clique_size(G)

        SHORTEST_PATH_LENGTHS = 0
        MEAN_SHORTEST_PATH_LENGTH = 0
        for shortest_paths in metas_df.loc[metas_df["idx"] == index]["shortest_paths"]:
            for shortest_path in shortest_paths:
                SHORTEST_PATH_LENGTHS += len(shortest_path)
            MEAN_SHORTEST_PATH_LENGTH += int(
                SHORTEST_PATH_LENGTHS / len(shortest_paths)
            )

        if candidates[-1] == "":

            site_link = float("NaN")
            outcoming = float("NaN")
            incoming = float("NaN")

        else:

            site_link = run_sitelinks(candidates[-1])
            outcoming = run_query_out(candidates[-1])
            incoming = run_query_in(candidates[-1])

        iter_data = [
            index,
            N_TRIANGLES,
            n_nodes,
            n_edges,
            N_KATZ_CENTRALITY,
            N_PAGERANK,
            n_largest_clique_size,
            SHORTEST_PATH_LENGTHS,
            MEAN_SHORTEST_PATH_LENGTH,
            site_link,
            outcoming,
            incoming,
            N_EIGENVECTOR_CENTRALITY,
            N_CLUSTERING,
        ]
        data.append(iter_data)

    header = [
        "index",
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
    ]

    with open("subgraph_data.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(data)
