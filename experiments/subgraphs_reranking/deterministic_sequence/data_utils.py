import ujson
import networkx as nx
import torch
import pandas as pd


def read_jsonl_to_pandas(data_path: str) -> pd.DataFrame:
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(ujson.loads(line))
    return pd.DataFrame(data)


def get_node_names(
    subgraph,
    candidate_start_token="[unused1]",
    candidate_end_token="[unused2]",
):
    node_names = [subgraph.nodes[node]["label"] for node in subgraph.nodes()]
    node_type = [subgraph.nodes[node]["type"] for node in subgraph.nodes()]

    if "ANSWER_CANDIDATE_ENTITY" in node_type:
        candidate_idx = node_type.index("ANSWER_CANDIDATE_ENTITY")
        node_names[
            candidate_idx
        ] = f"{candidate_start_token}{node_names[candidate_idx]}{candidate_end_token}"
    else:
        return None
    return node_names


def graph_to_sequence(subgraph, node_names):
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
            if edge_info == 0:  # no endge from_node -> to_node
                # sequence.extend([from_node, "None", to_node])
                pass
            else:
                sequence.extend([from_node, edge_info, to_node])
    sequence = ",".join(str(node) for node in sequence)
    return sequence


def linearize_graph(
    graph, candidate_start_token="[unused1]", candidate_end_token="[unused2]"
):
    graph_obj = nx.readwrite.json_graph.node_link_graph(graph)
    try:
        graph_node_names = get_node_names(
            graph_obj,
            candidate_start_token=candidate_start_token,
            candidate_end_token=candidate_end_token,
        )
        return graph_to_sequence(graph_obj, graph_node_names)
    except KeyError:
        return None
    except nx.NetworkXError:
        return None


def data_df_convert(
    df,
    candidate_start_token="[unused1]",
    candidate_end_token="[unused2]",
):
    # Filter all graphs without ANSWER_CANDIDATE_ENTITY
    df = df[df["graph"].apply(lambda x: "ANSWER_CANDIDATE_ENTITY" in str(x))]

    # Linearize graphs
    df.loc[:, "graph_sequence"] = df["graph"].apply(
        lambda graph: linearize_graph(graph, candidate_start_token, candidate_end_token)
    )
    df = df.dropna(subset=["graph_sequence"])

    # Preapre label (correct or not answer)
    # fmt: off
    df["correct"] = df.apply(
        lambda row: len(set(row["answerEntity"]).intersection(row["groundTruthAnswerEntity"])) > 0,
        axis=1,
    )
    # fmt: on

    return df


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = self.tokenizer(
            row["graph_sequence"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        item["input_ids"] = item["input_ids"].view(-1)
        item["attention_mask"] = item["attention_mask"].view(-1)
        item["labels"] = torch.tensor(row["correct"], dtype=torch.float)
        return item

    def __len__(self):
        return self.df.index.size
