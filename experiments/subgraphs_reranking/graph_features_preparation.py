""" prepare the graph features dataset from scratch"""
import argparse
from ast import literal_eval
import yaml
from datasets import load_dataset, Dataset, DatasetDict
from gtda.homology import FlagserPersistence
from gtda.graphs import GraphGeodesicDistance
from sentence_transformers import SentenceTransformer
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

parse = argparse.ArgumentParser()
parse.add_argument(
    "--subgraphs_dataset_path",
    default="s-nlp/Mintaka_Subgraphs_T5_xl_ssm",
    type=str,
    help="Path for the subgraphs dataset (HF)",
)

parse.add_argument(
    "--g2t_t5_train_path",
    type=str,
    default="/workspace/storage/misc/train_results_mintaka_T5XL.yaml",
    help="Path to g2t train yaml file",
)

parse.add_argument(
    "--g2t_t5_test_path",
    type=str,
    default="/workspace/storage/misc/test_results_mintaka_T5Large.yaml",
    help="Path to g2t test yaml file",
)

parse.add_argument(
    "--g2t_t5_val_path",
    type=str,
    default="/workspace/storage/misc/val_results_t5_xl.yaml",
    help="Path to g2t test yaml file",
)

parse.add_argument(
    "--g2t_gap_train_path",
    type=str,
    default="/workspace/storage/misc/gap_train_mintaka_large_predictions.txt",
    help="Path to g2t train yaml file",
)

parse.add_argument(
    "--g2t_gap_test_path",
    type=str,
    default="/workspace/storage/misc/gap_test_mintaka_large_predictions.txt",
    help="Path to g2t test yaml file",
)

parse.add_argument(
    "--g2t_gap_val_path",
    type=str,
    default="/workspace/storage/misc/gap_val_mintaka_t5_xl_filtered_predictions.txt",
    help="Path to g2t test yaml file",
)

parse.add_argument(
    "--upload_dataset",
    type=bool,
    default=True,
    help="whether to upload dataset to HuggingFace",
)

parse.add_argument(
    "--hf_path",
    type=str,
    default="hle2000/Mintaka_Graph_Features_T5-large-ssm",
    help="path to upload to HuggingFace",
)


def get_g2t_seqs(g2t_path):
    """proccess the g2t yaml file and return list of g2t seqs"""
    with open(g2t_path, "r", encoding="utf-8") as stream:
        try:
            g2t_seqs_raw = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    g2t_seqs = []
    for curr_seq in g2t_seqs_raw["data"]:
        g2t_seqs.append(curr_seq["predicted"])
    return g2t_seqs


def get_gap_seqs(gap_path):
    """proccess the gap txt file and return list of g2t seqs"""
    with open(gap_path, "r", encoding="utf-8") as file:
        gap_seqs = file.read().splitlines()
    return gap_seqs


def add_new_seqs(g2t_path, gap_path, dataframe):
    """get the new g2t and gap seqs and add to df"""
    g2t_seqs = get_g2t_seqs(g2t_path)
    gap_seqs = get_gap_seqs(gap_path)

    dataframe["g2t_sequence"] = g2t_seqs
    dataframe["gap_sequence"] = gap_seqs
    return dataframe


def get_distance_ans_cand(graph, ans_cand_id):
    """get avg distance from ans entity to answer candidate"""
    graph = graph.to_undirected()  # for ssp both ways
    ssp_dict = nx.shortest_path(graph, target=ans_cand_id)
    total_ssp, total_paths = 0, 0

    for cand, length in ssp_dict.items():
        if cand != ans_cand_id:
            total_ssp += len(length)
            total_paths += 1

    return total_ssp / total_paths


def get_graph_vector(heat, graph):
    """find tfidf vector of graph"""
    adj_matrix = nx.adjacency_matrix(graph)
    x_geo_dist = GraphGeodesicDistance(directed=True, unweighted=True).fit_transform(
        [adj_matrix]
    )
    persistance_diag = FlagserPersistence().fit_transform(x_geo_dist)
    scaled_pers_diag = np.nan_to_num(persistance_diag)
    heat_pers_diag = heat.fit_transform(scaled_pers_diag)
    return heat_pers_diag.ravel()


def find_label(graph, wd_id):
    """find label of the wikidata id using graph"""
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        if node["name_"] == wd_id:
            return node["label"]
    return f"cannot find label for {wd_id}"


def get_node_names(
    subgraph,
    candidate_start_token="[unused1]",
    candidate_end_token="[unused2]",
    highlight=False,
):
    """with graph, return the node names (if cand note, add token)"""
    node_names = [subgraph.nodes[node]["label"] for node in subgraph.nodes()]
    node_type = [subgraph.nodes[node]["type"] for node in subgraph.nodes()]

    if "ANSWER_CANDIDATE_ENTITY" not in node_type:
        return None

    if highlight:
        candidate_idx = node_type.index("ANSWER_CANDIDATE_ENTITY")
        node_names[
            candidate_idx
        ] = f"{candidate_start_token}{node_names[candidate_idx]}{candidate_end_token}"

    return node_names


def graph_to_sequence(subgraph, node_names):
    """original deterministic sequence"""
    # getting adjency matrix and weight info
    adj_matrix = nx.adjacency_matrix(subgraph).todense().tolist()
    edge_data = subgraph.edges.data()

    # adding our edge info
    for edge in edge_data:
        i, j, data = edge
        i, j = int(i), int(j)
        adj_matrix[i][j] = data["label"]

    sequence = []
    # for adjency matrix, i, j means node i -> j
    for i, row in enumerate(adj_matrix):
        from_node = node_names[i]  # from node (node i)
        for j, edge_info in enumerate(row):
            to_node = node_names[j]
            if edge_info != 0:  # no endge from_node -> to_node
                sequence.extend([from_node, edge_info, to_node])

    sequence = ",".join(str(node) for node in sequence)
    return sequence


def arr_to_str(arr):
    """array to str, seperated by comma"""
    arr = list(arr)
    return ",".join(str(a) for a in arr)


def try_literal_eval(strng):
    """str representation to object"""
    try:
        return literal_eval(strng)
    except ValueError:
        return strng


def find_candidate_note(graph):
    """find id of answer candidate node"""
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        if node["type"] == "ANSWER_CANDIDATE_ENTITY":
            return node_id
    raise ValueError("Cannot find answer candidate entity")


def get_features(dataframe, model, device):
    """get the graph features for the df"""
    dict_list = []
    for _, row in tqdm(dataframe.iterrows()):
        # convert from json dict to networkx graph
        graph_obj = json_graph.node_link_graph(try_literal_eval(row["graph"]))
        graph_node_names = get_node_names(graph_obj)
        g2t_seq = row["g2t_sequence"]
        gap_seq = row["gap_sequence"]

        # skip if we have no answer candidates in our graph
        try:
            ans_cand_id = find_candidate_note(graph_obj)
            ques_ans = (
                f"{row['question']} ; {find_label(graph_obj, row['answerEntity'])}"
            )
            determ_seq = graph_to_sequence(graph_obj, graph_node_names)

            # build the features
            curr_dict = {
                # text data
                "question": row["question"],
                "question_answer": ques_ans,
                # numerical data
                "num_nodes": graph_obj.number_of_nodes(),
                "num_edges": graph_obj.number_of_edges(),
                "density": nx.density(graph_obj),
                "cycle": len(nx.recursive_simple_cycles(graph_obj)),
                "bridge": len(
                    sorted(map(sorted, nx.k_edge_components(graph_obj, k=2)))
                ),
                "katz_centrality": nx.katz_centrality(graph_obj)[ans_cand_id],
                "page_rank": nx.pagerank(graph_obj)[ans_cand_id],
                "avg_ssp_length": get_distance_ans_cand(graph_obj, ans_cand_id),
                "determ_sequence": determ_seq,
                "gap_sequence": gap_seq,
                "g2t_sequence": g2t_seq,
                # embedding data
                "determ_sequence_embedding": arr_to_str(
                    model.encode(determ_seq, device=device, convert_to_numpy=True)
                ),
                "gap_sequence_embedding": arr_to_str(
                    model.encode(gap_seq, device=device, convert_to_numpy=True)
                ),
                "g2t_sequence_embedding": arr_to_str(
                    model.encode(g2t_seq, device=device, convert_to_numpy=True)
                ),
                "question_answer_embedding": arr_to_str(
                    model.encode(ques_ans, device=device, convert_to_numpy=True)
                ),
                "tfidf_vector": np.array([]),
                # label
                "correct": float(row["correct"]),
            }
        except:  # pylint: disable=bare-except
            continue
        dict_list.append(curr_dict)

    final_df = pd.DataFrame(dict_list)
    return final_df


if __name__ == "__main__":
    args = parse.parse_args()

    subgraphs_dataset = load_dataset(
        args.subgraphs_dataset_path, cache_dir="/workspace/storage/misc/huggingface"
    )
    train_df = subgraphs_dataset["train"].to_pandas()
    val_df = subgraphs_dataset["validation"].to_pandas()
    test_df = subgraphs_dataset["test"].to_pandas()

    # adding the new g2t sequences to subgraph dataset
    train_df = add_new_seqs(args.g2t_t5_train_path, args.g2t_gap_train_path, train_df)
    test_df = add_new_seqs(args.g2t_t5_test_path, args.g2t_gap_test_path, test_df)
    val_df = add_new_seqs(args.g2t_t5_val_path, args.g2t_gap_val_path, val_df)

    # get all features and add to df
    smodel = SentenceTransformer("all-mpnet-base-v2")
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_train_df = get_features(train_df, smodel, curr_device)
    processed_test_df = get_features(test_df, smodel, curr_device)
    processed_val_df = get_features(val_df, smodel, curr_device)

    # upload to HF
    if args.upload_dataset:
        ds = DatasetDict()
        ds["train"] = Dataset.from_pandas(processed_train_df)
        ds["validation"] = Dataset.from_pandas(processed_val_df)
        ds["test"] = Dataset.from_pandas(processed_test_df)
        ds.push_to_hub(args.hf_path)
