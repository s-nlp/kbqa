"""data utils for graphormer"""
from ast import literal_eval
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.models.graphormer.collating_graphormer import algos_graphormer


def transform_graph(graph_data, answer_entity, ground_truth_entity):
    """transform graph to graphormer's formatted graph"""
    # Create an empty dictionary to store the transformed graph
    transformed_graph = {}

    # Extract 'nodes' and 'links' from the graph_data
    nodes = graph_data["nodes"]
    links = graph_data["links"]

    # Calculate num_nodes
    num_nodes = len(nodes)

    # Calculate edge_index
    edge_index = [[link["source"], link["target"]] for link in links]
    edge_index = list(zip(*edge_index))

    # Check if "answerEntity" matches with "groundTruthAnswerEntity" to get the label (y)
    label = 1.0 if answer_entity in ground_truth_entity else 0.0

    # Calculate node_feat based on 'type' key
    node_feat = []
    for node in nodes:
        if node["type"] == "INTERNAL":
            node_feat.append([1])
        elif node["type"] == "ANSWER_CANDIDATE_ENTITY":
            node_feat.append([2])
        elif node["type"] == "QUESTIONS_ENTITY":
            node_feat.append([3])

    # Store the calculated values in the transformed_graph dictionary
    transformed_graph["edge_index"] = edge_index
    transformed_graph["num_nodes"] = num_nodes
    transformed_graph["y"] = [label]
    transformed_graph["node_feat"] = node_feat
    transformed_graph["edge_attr"] = [[0]]

    return transformed_graph


def create_adjacency_matrix(edge_list):
    """from edge list, create the adjacency matrix"""
    # Find the maximum node ID in the edge_list
    max_node_id = max(max(edge_list[0]), max(edge_list[1]))

    # Initialize an empty adjacency matrix with zeros
    adjacency_matrix = np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.int32)

    # Add edges to the adjacency matrix
    for src, dest in zip(edge_list[0], edge_list[1]):
        adjacency_matrix[src, dest] = 1

    return adjacency_matrix


def preprocess(item):
    """Convert to the required format for Graphormer"""
    attn_edge_type = None  # Initialize outside the loop

    # Calculate adjacency matrix
    adj = create_adjacency_matrix(item["edge_index"])

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)

    try:
        # Calculate max_dist and input_edges if the function call succeeds
        shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
        max_dist = np.amax(shortest_path_result)
        attn_edge_type = np.zeros(
            (item["num_nodes"], item["num_nodes"], len(item["edge_attr"])),
            dtype=np.int64,
        )
        input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    except:  # pylint: disable=bare-except
        # If the function call fails, handle the exception
        max_dist = 0
        attn_edge_type = None
        input_edges = np.zeros(
            (item["num_nodes"], item["num_nodes"], max_dist, len(item["edge_attr"])),
            dtype=np.int64,
        )
        shortest_path_result = None

    if attn_edge_type is None:
        # Initialize attn_edge_type here if it hasn't been initialized already
        attn_edge_type = np.zeros(
            (item["num_nodes"], item["num_nodes"], len(item["edge_attr"])),
            dtype=np.int64,
        )

    # Set values for all the keys
    processed_item = {
        "edge_index": np.array(item["edge_index"]),
        "num_nodes": item["num_nodes"],
        "y": item["y"],
        "node_feat": np.array(item["node_feat"]),
        "input_nodes": np.array(
            item["node_feat"]
        ),  # Use node_feat as input_nodes if node_feat is the feature representation
        "edge_attr": np.array(item["edge_attr"]),
        "attn_bias": np.zeros(
            (item["num_nodes"] + 1, item["num_nodes"] + 1), dtype=np.single
        ),
        "attn_edge_type": attn_edge_type,
        "spatial_pos": shortest_path_result.astype(np.int64) + 1,
        "in_degree": np.sum(adj, axis=1).reshape(-1) + 1,
        "out_degree": np.sum(adj, axis=1).reshape(-1) + 1,  # for undirected graph
        "input_edges": input_edges + 1,
        "labels": item.get(
            "labels", item["y"]
        ),  # Assuming "labels" key may or may not exist in the input data
    }

    return processed_item


def try_literal_eval(dict_str):
    """convert from str dict to dict type"""
    try:
        return literal_eval(dict_str)
    except ValueError:
        return dict_str


def transform_data(dataframe, save_path):
    """given data in df format, transform to graphormer format"""
    transformed_graph_dicts = []
    for _, row in tqdm(
        dataframe.iterrows(), total=len(dataframe), desc="Transforming graphs"
    ):
        try:
            curr_dict = {}
            graph_data = try_literal_eval(row["graph"])  # convert to dict
            curr_dict["original_graph"] = graph_data

            transformed_graph = transform_graph(
                graph_data, row["answerEntity"], row["groundTruthAnswerEntity"]
            )
            if (
                len(transformed_graph["edge_index"][0])
                or len(transformed_graph["edge_index"][1]) > 1
            ):
                curr_dict["question"] = row["question"]
                curr_dict["answerEntity"] = row["answerEntity"]
                curr_dict["groundTruthAnswerEntity"] = row["groundTruthAnswerEntity"]
                curr_dict["correct"] = row["correct"]
                curr_dict["transformed_graph"] = transformed_graph
                transformed_graph_dicts.append(curr_dict)
        except:  # pylint: disable=bare-except
            continue

    # save to save_path if not none
    if save_path:
        with open(save_path, "w+", encoding="utf-8") as file:
            for transformed_graph in transformed_graph_dicts:
                file.write(json.dumps(transformed_graph) + "\n")
    return transformed_graph_dicts


class GraphDataset(Dataset):
    """custom graph dataset for graphormer"""

    def __init__(self, transformed_data):
        self.data = []

        for graph_dict in transformed_data:
            preproc_graph = preprocess(graph_dict["transformed_graph"])

            if preproc_graph["input_edges"].shape[2] != 0:
                self.data.append(preproc_graph)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
