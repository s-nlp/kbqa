"""
prepare sequence data from scratch and upload to HF
"""
from ast import literal_eval
import argparse
import torch
from datasets import load_dataset, Dataset, DatasetDict
import networkx as nx
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import yaml
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument(
    "model_name",
    type=str,
    default="sentence-transformers/all-mpnet-base-v2",
    help="HF model name for AutoModelForSequenceClassification",
)

parse.add_argument(
    "--data_path",
    type=str,
    default="hle2000/Mintaka_Subgraphs_T5_xl_ssm",
    help="Path to train JSONL file",
)

parse.add_argument(
    "--g2t_train_path",
    type=str,
    default="/workspace/storage/misc/train_results_mintaka_xl.yaml",
    help="Path to g2t train yaml file",
)

parse.add_argument(
    "--g2t_test_path",
    type=str,
    default="/workspace/storage/misc/test_results_mintaka.yaml",
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
    default="hle2000/test",
    help="path to upload to HuggingFace",
)


def tokenizer_model(model_name, dev):
    """return tokenizer w/ tokens"""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
    mod = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(dev)
    return tok, mod


def add_new_seqs(path, dataframe):
    """get the new seqs from yaml and add to df"""
    with open(path, "r", encoding="utf-8") as stream:
        try:
            new_seqs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    updated_seqs = []
    for curr_seq in new_seqs["data"]:
        updated_seqs.append(curr_seq["predicted"])
    dataframe["updated_sequence"] = updated_seqs
    return dataframe


def get_node_names(
    subgraph,
    candidate_start_token="[unused1]",
    candidate_end_token="[unused2]",
    highlight=False,
):
    """get nodes name of the subgraph, pad candidates with token"""
    node_names = [subgraph.nodes[node]["label"] for node in subgraph.nodes()]
    node_type = [subgraph.nodes[node]["type"] for node in subgraph.nodes()]

    if highlight:
        candidate_idx = node_type.index("ANSWER_CANDIDATE_ENTITY")
        node_names[
            candidate_idx
        ] = f"{candidate_start_token}{node_names[candidate_idx]}{candidate_end_token}"

    return node_names


def find_label(graph, wd_id):
    """find label of the wikidata id using graph"""
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        if node["name_"] == wd_id:
            return node["label"]
    return f"cannot find label for {wd_id}"


def try_literal_eval(str_dict):
    """turn str dict into dict type"""
    try:
        return literal_eval(str_dict)
    except ValueError:
        return str_dict


def graph_to_sequence(subgraph, node_names):
    """convert graph to sequence"""
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
            sequence.extend([from_node, edge_info, to_node])
    sequence = ",".join(str(node) for node in sequence)
    return sequence


def get_sequences(
    dataframe,
    tok,
    candidate_start_token,
    candidate_end_token,
):
    """get the sequences"""
    questions = list(dataframe["question"])
    graphs = list(dataframe["graph"])
    hl_graph_seq, no_hl_graph_seq = [], []

    for question, graph in zip(questions, graphs):
        graph_obj = nx.readwrite.json_graph.node_link_graph(try_literal_eval(graph))
        try:
            hl_graph_node_names = get_node_names(
                graph_obj, candidate_start_token, candidate_end_token, highlight=True
            )
            hl_seq = graph_to_sequence(graph_obj, hl_graph_node_names)

            no_hl_graph_node_names = get_node_names(
                graph_obj, candidate_start_token, candidate_end_token, highlight=False
            )
            no_hl_seq = graph_to_sequence(graph_obj, no_hl_graph_node_names)

            hl_seq = f"{question}{tok.sep_token}{hl_seq}"
            no_hl_seq = f"{question}{tok.sep_token}{no_hl_seq}"
        except KeyError:
            hl_seq, no_hl_seq = None, None
        except nx.NetworkXError:
            hl_seq, no_hl_seq = None, None
        hl_graph_seq.append(hl_seq)
        no_hl_graph_seq.append(no_hl_seq)

    return no_hl_graph_seq, hl_graph_seq


def preproc_updated_sequences(
    dataframe, tok, candidate_start_token, candidate_end_token
):
    """return highlighted and non-highlighted sequence"""
    no_hl_seqs, hl_seqs = [], []
    for _, group in tqdm(dataframe.iterrows()):
        graph_obj = nx.readwrite.json_graph.node_link_graph(
            try_literal_eval(group["graph"])
        )
        updated_seq = group["updated_sequence"]

        try:
            ans_ent_label = find_label(graph_obj, group["answerEntity"])
            splits = updated_seq.split(ans_ent_label)
            hl_seq = f"{splits[0].strip()} {candidate_start_token}{ans_ent_label}\
                {candidate_end_token} {splits[1].strip()}"
            hl_seq = f"{group['question']}{tok.sep_token}{hl_seq}"

            no_hl_seq = f"{group['question']}{tok.sep_token}{updated_seq}"
        except:  # pylint: disable=bare-except
            hl_seq, no_hl_seq = None, None
        no_hl_seqs.append(no_hl_seq)
        hl_seqs.append(hl_seq)
    return no_hl_seqs, hl_seqs


def data_df_convert(dataframe, tok, candidate_start_token, candidate_end_token):
    """given the df, add the sequences and remove unnecessary info"""
    # Filter all graphs without ANSWER_CANDIDATE_ENTITY
    dataframe = dataframe[
        dataframe["graph"].apply(lambda x: "ANSWER_CANDIDATE_ENTITY" in str(x))
    ]

    # processing the updated g2t sequence
    no_hl_seqs, hl_seqs = preproc_updated_sequences(
        dataframe, tok, candidate_start_token, candidate_end_token
    )
    dataframe["highlighted_updated_sequence"] = hl_seqs
    dataframe["no_highlighted_updated_sequence"] = no_hl_seqs

    # processing the determ sequence
    no_hl_graph_seq, hl_graph_seq = get_sequences(
        dataframe, tok, candidate_start_token, candidate_end_token
    )
    dataframe["highlighted_sequence"] = hl_graph_seq
    dataframe["no_highlighted_sequence"] = no_hl_graph_seq

    # filter out all invalid entries
    dataframe = dataframe.dropna(
        subset=[
            "highlighted_updated_sequence",
            "no_highlighted_updated_sequence",
            "highlighted_sequence",
            "no_highlighted_sequence",
        ]
    )
    return dataframe


if __name__ == "__main__":
    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = tokenizer_model(args.model_name, device)

    # getting subgraph dataset
    subgraphs_dataset = load_dataset(args.data_path)
    train_df = subgraphs_dataset["train"].to_pandas()
    test_df = subgraphs_dataset["test"].to_pandas()

    # adding the new g2t sequences to subgraph dataset
    train_df = add_new_seqs(args.g2t_train_path, train_df)
    test_df = add_new_seqs(args.g2t_test_path, test_df)

    # getting the final dataset
    CANDIDATE_START_TOKEN = "[unused1]"
    CANDIDATE_END_TOKEN = "[unused2]"
    train_df = data_df_convert(
        train_df, tokenizer, CANDIDATE_START_TOKEN, CANDIDATE_END_TOKEN
    )
    test_df = data_df_convert(
        test_df, tokenizer, CANDIDATE_START_TOKEN, CANDIDATE_END_TOKEN
    )
    # upload to HF
    if args.upload_dataset:
        ds = DatasetDict()
        ds["train"] = Dataset.from_pandas(train_df)
        ds["test"] = Dataset.from_pandas(test_df)
        ds.push_to_hub(args.hf_path)
