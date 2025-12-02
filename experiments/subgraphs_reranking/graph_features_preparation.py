""" prepare the graph features dataset from scratch"""
import argparse
import json
import os
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
    default=None,
    type=str,
    help="Path for the subgraphs dataset (HF). Required if not using JSON files.",
)

parse.add_argument(
    "--subgraphs_train_path",
    type=str,
    default=None,
    help="Path to train JSON/JSONL file (optional)",
)

parse.add_argument(
    "--subgraphs_val_path",
    type=str,
    default=None,
    help="Path to validation JSON/JSONL file (optional)",
)

parse.add_argument(
    "--subgraphs_test_path",
    type=str,
    default=None,
    help="Path to test JSON/JSONL file (optional)",
)

parse.add_argument(
    "--g2t_t5_train_path",
    type=str,
    default=None,
    help="Path to g2t train yaml file",
)

parse.add_argument(
    "--g2t_t5_test_path",
    type=str,
    default=None,
    help="Path to g2t test yaml file",
)

parse.add_argument(
    "--g2t_t5_val_path",
    type=str,
    default=None,
    help="Path to g2t val yaml file",
)

parse.add_argument(
    "--g2t_gap_train_path",
    type=str,
    default=None,
    help="Path to g2t gap train txt file",
)

parse.add_argument(
    "--g2t_gap_test_path",
    type=str,
    default=None,
    help="Path to g2t gap test txt file",
)

parse.add_argument(
    "--g2t_gap_val_path",
    type=str,
    default=None,
    help="Path to g2t gap val txt file",
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

parse.add_argument(
    "--g2t_types",
    type=str,
    nargs="+",
    default=["determ", "t5", "gap"],
    choices=["determ", "t5", "gap"],
    help="G2T types to process: determ (G2T Deterministic), t5 (G2T T5), gap (G2T GAP). "
    "When any G2T type is selected, G2T Deterministic is always included.",
)

parse.add_argument(
    "--subset_name",
    type=str,
    default="mkqa_t5large",
    help="Name for the subset when pushing to HuggingFace. Subset will be named 'name_subgraphs'.",
)


def load_json_dataset(json_path):
    """load dataset from JSON/JSONL file"""
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Compute 'correct' field if missing
    if "correct" not in df.columns:
        if "groundTruthAnswerEntity" in df.columns and "answerEntity" in df.columns:
            def compute_correct(row):
                answer_entity = row["answerEntity"]
                if isinstance(answer_entity, str):
                    answer_entity = [answer_entity]
                elif not isinstance(answer_entity, list):
                    answer_entity = [str(answer_entity)]
                ground_truth = row["groundTruthAnswerEntity"]
                if isinstance(ground_truth, str):
                    ground_truth = [ground_truth]
                elif not isinstance(ground_truth, list):
                    ground_truth = [str(ground_truth)]
                return 1.0 if any(str(ans) in ground_truth for ans in answer_entity) else 0.0
            df["correct"] = df.apply(compute_correct, axis=1)
        else:
            df["correct"] = 0.0
    
    # Convert graph to string if it's a dict
    if "graph" in df.columns and df["graph"].dtype == "object":
        df["graph"] = df["graph"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    
    return df


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


def add_new_seqs(g2t_t5_path, g2t_gap_path, dataframe, g2t_types):
    """get the new g2t and gap seqs and add to df based on selected g2t_types"""
    if "t5" in g2t_types:
        if g2t_t5_path is None:
            raise ValueError("G2T T5 path is required when processing G2T T5 sequences")
        g2t_seqs = get_g2t_seqs(g2t_t5_path)
        dataframe["g2t_sequence"] = g2t_seqs
    if "gap" in g2t_types:
        if g2t_gap_path is None:
            raise ValueError("G2T GAP path is required when processing G2T GAP sequences")
        gap_seqs = get_gap_seqs(g2t_gap_path)
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


def parse_graph(graph_data):
    """parse graph data from dict, JSON string, or Python literal string"""
    if isinstance(graph_data, dict):
        return graph_data
    if isinstance(graph_data, str):
        try:
            return json.loads(graph_data)
        except (json.JSONDecodeError, ValueError):
            try:
                return literal_eval(graph_data)
            except (ValueError, SyntaxError):
                return graph_data
    return graph_data


def find_candidate_note(graph):
    """find id of answer candidate node"""
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        if node["type"] == "ANSWER_CANDIDATE_ENTITY":
            return node_id
    raise ValueError("Cannot find answer candidate entity")


def get_features(dataframe, model, device, g2t_types):
    """get the graph features for the df based on selected g2t_types"""
    # Identify fields to preserve from original dataframe
    fields_to_preserve = ["answerEntity", "groundTruthAnswerEntity", "questionEntity", "graph", "correct"]
    available_fields = [f for f in fields_to_preserve if f in dataframe.columns]
    
    dict_list = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing rows"):
        # convert from json dict to networkx graph
        graph_data = parse_graph(row["graph"])
        graph_obj = json_graph.node_link_graph(graph_data, edges="links")
        graph_node_names_no_highlight = get_node_names(graph_obj, highlight=False)
        graph_node_names_highlight = get_node_names(graph_obj, highlight=True)

        # skip if we have no answer candidates in our graph
        try:
            ans_cand_id = find_candidate_note(graph_obj)
            # Get answerEntity - handle both list and single value
            answer_entity = row.get("answerEntity", "")
            if isinstance(answer_entity, list) and len(answer_entity) > 0:
                answer_entity = answer_entity[0]
            ques_ans = (
                f"{row['question']} ; {find_label(graph_obj, answer_entity)}"
            )

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
            }
            
            # Always preserve original fields from JSON (answerEntity, groundTruthAnswerEntity, questionEntity, graph, correct)
            for field in available_fields:
                if field in row:
                    if field == "graph":
                        # Preserve graph in original format (should be string after load_json_dataset)
                        graph_original = row["graph"]
                        if isinstance(graph_original, dict):
                            # If still a dict, convert to JSON string
                            curr_dict["graph"] = json.dumps(graph_original)
                        elif isinstance(graph_original, str):
                            # Already a string, preserve as-is
                            curr_dict["graph"] = graph_original
                        else:
                            # Fallback: convert to string
                            curr_dict["graph"] = str(graph_original)
                    elif field == "correct":
                        curr_dict["correct"] = float(row["correct"])
                    else:
                        # Preserve answerEntity, groundTruthAnswerEntity, questionEntity as-is
                        curr_dict[field] = row[field]

            # Process G2T Deterministic - generate both highlighted and no_highlighted versions
            if "determ" in g2t_types:
                # No highlighted version
                no_highlighted_determ_seq = graph_to_sequence(graph_obj, graph_node_names_no_highlight)
                curr_dict["no_highlighted_determ_sequence"] = no_highlighted_determ_seq
                curr_dict["no_highlighted_determ_sequence_embedding"] = arr_to_str(
                    model.encode(no_highlighted_determ_seq, device=device, convert_to_numpy=True)
                )
                
                # Highlighted version
                highlighted_determ_seq = graph_to_sequence(graph_obj, graph_node_names_highlight)
                curr_dict["highlighted_determ_sequence"] = highlighted_determ_seq
                curr_dict["highlighted_determ_sequence_embedding"] = arr_to_str(
                    model.encode(highlighted_determ_seq, device=device, convert_to_numpy=True)
                )

            # Process G2T T5
            if "t5" in g2t_types:
                g2t_seq = row["g2t_sequence"]
                curr_dict["g2t_sequence"] = g2t_seq
                curr_dict["g2t_sequence_embedding"] = arr_to_str(
                    model.encode(g2t_seq, device=device, convert_to_numpy=True)
                )

            # Process G2T GAP
            if "gap" in g2t_types:
                gap_seq = row["gap_sequence"]
                curr_dict["gap_sequence"] = gap_seq
                curr_dict["gap_sequence_embedding"] = arr_to_str(
                    model.encode(gap_seq, device=device, convert_to_numpy=True)
                )

            # Always include question_answer embedding
            curr_dict["question_answer_embedding"] = arr_to_str(
                model.encode(ques_ans, device=device, convert_to_numpy=True)
            )

        except:  # pylint: disable=bare-except
            continue
        dict_list.append(curr_dict)

    final_df = pd.DataFrame(dict_list)
    return final_df


if __name__ == "__main__":
    args = parse.parse_args()
    g2t_types = args.g2t_types

    # Always include determ when any G2T type is selected
    if "determ" not in g2t_types and len(g2t_types) > 0:
        g2t_types = ["determ"] + g2t_types

    # Load dataset from JSON files or HuggingFace
    train_df = None
    val_df = None
    test_df = None

    if args.subgraphs_train_path or args.subgraphs_val_path or args.subgraphs_test_path:
        # Load from JSON files
        if args.subgraphs_train_path:
            if not os.path.exists(args.subgraphs_train_path):
                raise ValueError(f"Train JSON file not found: {args.subgraphs_train_path}")
            train_df = load_json_dataset(args.subgraphs_train_path)
        
        if args.subgraphs_val_path:
            if not os.path.exists(args.subgraphs_val_path):
                raise ValueError(f"Validation JSON file not found: {args.subgraphs_val_path}")
            val_df = load_json_dataset(args.subgraphs_val_path)
        
        if args.subgraphs_test_path:
            if not os.path.exists(args.subgraphs_test_path):
                raise ValueError(f"Test JSON file not found: {args.subgraphs_test_path}")
            test_df = load_json_dataset(args.subgraphs_test_path)
    elif args.subgraphs_dataset_path:
        # Load from HuggingFace
        subgraphs_dataset = load_dataset(
            args.subgraphs_dataset_path, cache_dir="/workspace/storage/misc/huggingface"
        )
        train_df = subgraphs_dataset["train"].to_pandas()
        val_df = subgraphs_dataset["validation"].to_pandas()
        test_df = subgraphs_dataset["test"].to_pandas()
    else:
        raise ValueError(
            "Either --subgraphs_dataset_path (HF) or at least one of "
            "--subgraphs_train_path/--subgraphs_val_path/--subgraphs_test_path (JSON) must be provided"
        )

    # adding the new g2t sequences to subgraph dataset
    if train_df is not None:
        train_df = add_new_seqs(
            args.g2t_t5_train_path, args.g2t_gap_train_path, train_df, g2t_types
        )
    if test_df is not None:
        test_df = add_new_seqs(
            args.g2t_t5_test_path, args.g2t_gap_test_path, test_df, g2t_types
        )
    if val_df is not None:
        val_df = add_new_seqs(
            args.g2t_t5_val_path, args.g2t_gap_val_path, val_df, g2t_types
        )

    # get all features and add to df
    smodel = SentenceTransformer("all-mpnet-base-v2")
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processed_train_df = None
    processed_test_df = None
    processed_val_df = None
    
    if train_df is not None:
        processed_train_df = get_features(train_df, smodel, curr_device, g2t_types)
    if test_df is not None:
        processed_test_df = get_features(test_df, smodel, curr_device, g2t_types)
    if val_df is not None:
        processed_val_df = get_features(val_df, smodel, curr_device, g2t_types)

    # upload to HF
    if args.upload_dataset:
        ds = DatasetDict()
        if processed_train_df is not None:
            ds["train"] = Dataset.from_pandas(processed_train_df)
        if processed_val_df is not None:
            ds["validation"] = Dataset.from_pandas(processed_val_df)
        if processed_test_df is not None:
            ds["test"] = Dataset.from_pandas(processed_test_df)
        if len(ds) > 0:
            subset_name = f"{args.subset_name}_subgraphs"
            # Push dataset with config_name to create a subset/configuration
            # Can be loaded later with: load_dataset(hf_path, subset_name)
            try:
                ds.push_to_hub(args.hf_path, config_name=subset_name)
            except (TypeError, ValueError):
                # If config_name is not supported with DatasetDict, push normally
                # Note: subset organization may need to be handled differently
                ds.push_to_hub(args.hf_path)
                print(f"Note: Pushed dataset without config_name. Subset name '{subset_name}' is for reference only.")
