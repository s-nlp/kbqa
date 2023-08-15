""" data utils function for sequences reranking """
from ast import literal_eval
import networkx as nx
import torch


def get_node_names(
    subgraph,
    candidate_start_token="[unused1]",
    candidate_end_token="[unused2]",
):
    """get nodes name of the subgraph, pad candidates with token"""
    node_names = [subgraph.nodes[node]["label"] for node in subgraph.nodes()]
    node_type = [subgraph.nodes[node]["type"] for node in subgraph.nodes()]

    candidate_idx = node_type.index("ANSWER_CANDIDATE_ENTITY")
    node_names[
        candidate_idx
    ] = f"{candidate_start_token}{node_names[candidate_idx]}{candidate_end_token}"

    return node_names


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


def try_literal_eval(str_dict):
    """turn str dict into dict type"""
    try:
        return literal_eval(str_dict)
    except ValueError:
        return str_dict


def get_sequences(
    dataframe, tokenizer, candidate_start_token, candidate_end_token, context=True
):
    """get the sequences"""
    questions = list(dataframe["question"])
    graphs = list(dataframe["graph"])
    graph_seq = []

    for question, graph in zip(questions, graphs):
        graph_obj = nx.readwrite.json_graph.node_link_graph(try_literal_eval(graph))
        try:
            graph_node_names = get_node_names(
                graph_obj, candidate_start_token, candidate_end_token
            )
            curr_seq = graph_to_sequence(graph_obj, graph_node_names)
            if context:
                curr_seq = f"{question}{tokenizer.sep_token}{curr_seq}"
        except KeyError:
            curr_seq = None
        except nx.NetworkXError:
            curr_seq = None
        graph_seq.append(curr_seq)

    return graph_seq


def data_df_convert(
    dataframe, tokenizer, candidate_start_token, candidate_end_token, context
):
    """given the df, add the sequences and remove unnecessary info"""
    # Filter all graphs without ANSWER_CANDIDATE_ENTITY
    dataframe = dataframe[
        dataframe["graph"].apply(lambda x: "ANSWER_CANDIDATE_ENTITY" in str(x))
    ]

    # get the sequences
    df_sequences = get_sequences(
        dataframe, tokenizer, candidate_start_token, candidate_end_token, context
    )
    dataframe["graph_sequence"] = df_sequences
    dataframe = dataframe.dropna(subset=["graph_sequence"])  # remove faulty seqs

    return dataframe


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset class for sequences"""

    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        item = self.tokenizer(
            row["graph_sequence"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        item["input_ids"] = item["input_ids"].view(-1)
        item["attention_mask"] = item["attention_mask"].view(-1)
        item["labels"] = torch.tensor(  # pylint: disable=no-member
            row["correct"], dtype=torch.float  # pylint: disable=no-member
        )
        return item

    def __len__(self):
        return self.dataframe.index.size
