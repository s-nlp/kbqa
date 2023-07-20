"""
Module to add weights to the subgraphs by similarity using PBG embeddings
"""

import argparse
import numpy as np
import pandas as pd
from multiprocessing import JoinableQueue, Process
from queue import Empty as QueueEmpty


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pbg_tsv",
    default="/mnt/raid/pbg.tsv",
    type=str,
)

parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
)

parser.add_argument(
    "--entities_json",
    default="/mnt/raid/entities.json",
    type=str,
)

parser.add_argument(
    "--subgraphs_jsonl",
    default="/mnt/storage/QA_System_Project/subgrraphs_dataset/MINTAKA/mintaka_test_labeled.jsonl",
    type=str,
)

parser.add_argument(
    "--subgraphs_weighted_jsonl",
    default="/workspace/subgraphs/MINTAKA/mintaka_test_weighted_labeled.jsonl",
    type=str,
)


def find_substring(string, dataframe):
    """
    function to search substring for entity in pbg tsv
    """
    string = "<http://www.wikidata.org/entity/" + string + ">"
    mask = dataframe.iloc[:, 0].str.contains(string)
    target = dataframe[mask]
    return np.array(target.iloc[:, 1:]).flatten()


def read_jsonl_file(file_path):
    """
    reading jsonl file into dataframe
    """
    jsonl_df = pd.read_json(file_path, lines=True)
    return jsonl_df


def process_graph(subgraph, find_substring, dataframe):
    """
    function to store node embeddings and link weights for the subgraphs
    """
    try:
        # Iterate over each node
        for node in subgraph["nodes"]:
            # Extract the node's name_
            node_name = node["name_"]

            # Call the find_substring function to get the vector embedding
            embedding = find_substring(node_name, dataframe)

            # Assign the embedding to the node
            node["embedding"] = embedding.tolist()

        # Iterate over each link/edge
        for link in subgraph["links"]:
            # Get the source and target node indices
            source_index = link["source"]
            target_index = link["target"]

            # Get the embeddings of the source and target nodes
            source_embedding = subgraph["nodes"][source_index]["embedding"]
            target_embedding = subgraph["nodes"][target_index]["embedding"]

            # Calculate the dot product of the source and target embeddings
            dot_product = np.dot(source_embedding, target_embedding)

            # Assign the dot product as the weight of the edge in a normalized way
            link["weight"] = dot_product / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding)
            )

    except Exception as exception:
        print(f"An error occurred: {str(exception)}")

    return subgraph


def process_and_update_subgraphs(subgraphs_queue, result_df, processed_queue):
    """
    Function to process the subgraphs and update them
    """
    while True:
        try:
            subgraph = subgraphs_queue.get(1)
            processed_graph = process_graph(
                subgraph, find_substring, result_df
            )
            processed_queue.put((subgraph, processed_graph))
            subgraphs_queue.task_done()
        except QueueEmpty:
            break


def process_subgraph_queue(subgraph_queue, result_df, processed_queue):
    """
    function to store embeddings
    """
    while True:
        try:
            subgraph = subgraph_queue.get(1)
            process_and_update_subgraphs(subgraph, result_df, processed_queue)
        except QueueEmpty:
            break


if __name__ == "__main__":
    args = parser.parse_args()

    embeddings = np.loadtxt(
        args.pbg_tsv,
        dtype=np.float32,
        delimiter="\t",
        skiprows=1,
        usecols=range(1, 201),
        comments=None,
    )

    # Specify the path to the JSON file
    json_file_path = args.entities_json
    entities_df = pd.read_json(json_file_path)

    # Convert the NumPy array to a DataFrame
    embeddings_array_df = pd.DataFrame(embeddings)

    # Concatenate the DataFrame and the array vertically
    result = pd.concat([entities_df, embeddings_array_df], axis=1)

    subgraphs_file_path = args.subgraphs_jsonl
    test_graphs = read_jsonl_file(subgraphs_file_path)

    # Create a Queue to hold subgraphs and a Queue to hold processed subgraphs
    subgraph_queue = JoinableQueue()
    processed_queue = JoinableQueue()

    # Enqueue the subgraphs into the subgraph_queue
    for index, row in test_graphs.iterrows():
        subgraph_queue.put({"index": index, "graph": row["graph"]})

    num_processes = args.num_workers
    processes = []

    # Start worker processes
    for _ in range(num_processes):
        p = Process(
            target=process_subgraph_queue,
            args=(subgraph_queue, result, processed_queue),
        )
        p.start()
        processes.append(p)

    # Wait for all subgraphs to be processed
    subgraph_queue.join()

    # Stop worker processes
    for p in processes:
        p.terminate()

    # Retrieve processed subgraphs from the processed_queue and update test_graphs DataFrame
    while not processed_queue.empty():
        index, processed_graph = processed_queue.get(1)
        test_graphs.at[index, "graph"] = processed_graph

    # Define the path to the JSONL file
    output_file = args.subgraphs_weighted_jsonl
    test_graphs.to_json(output_file, orient="records", lines=True)

    print(f"DataFrame saved as JSONL file: {output_file}")
