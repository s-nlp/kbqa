"""_summary_
"""
from __future__ import annotations
import argparse
import enum
import os
import time
from datetime import datetime
from multiprocessing import JoinableQueue, Process
from pathlib import Path
from queue import Empty as QueueEmpty
from joblib import Memory
import igraph as ig
import networkx as nx
import psutil
import ujson


parse = argparse.ArgumentParser()
parse.add_argument(
    "--save_jsonl_path",
    default="/workspace/storage/new_subgraph_dataset/t5-xl-ssm/mintaka_test.jsonl",
    type=str,
    help="Path to resulting JSONL: subgraphs_dataset_prepared_entities_jsonl_path and graph",
)
parse.add_argument(
    "--igraph_wikidata_path",
    default="/workspace/storage/igraph_parsed_kg/wikidata_igraph_v2/wikidata_triples.txt",
    type=str,
    help="Path to Ncol or Pkl file with WD whole graph",
)
parse.add_argument(
    "--subgraphs_dataset_prepared_entities_jsonl_path",
    default="/workspace/storage/mintaka_seq2seq/t5-xl-ssm/train/mintaka_train.jsonl",
    type=str,
    help="Path to file with prepared entities with questions from dataset)",
)
parse.add_argument(
    "--n_jobs",
    type=int,
    default=32,
    help="Number of parallel process for ssp. ATTENTION: Each process require ~60-80Gb RAM",
)
parse.add_argument(
    "--skip_lines",
    type=int,
    default=0,
    help="Number of lines for skip in prepared_entities_jsonl file",
)

memory = Memory("/tmp/mem", verbose=0)


def now():
    """gets the current time"""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# pylint: disable=too-few-public-methods
class BColors:
    """colors for pretty printout"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class SubgraphNodeType(str, enum.Enum):
    """SubgraphNodeType - Enum class with types of subgraphs nodes"""

    INTERNAL = "Internal"
    QUESTIONS_ENTITY = "Question entity"
    QUESTIONS_ENTITY_NEIGHBOUR = "Question entity neighbour"
    ANSWER_CANDIDATE_ENTITY = "Answer candidate entity"


def extract_subgraph(
    wd_graph: ig.Graph,
    answer_entities: list[str],
    question_entities: list[str],
) -> ig.Graph:
    """given a list of question entities and answer candidates,
    extract an igraph subgraph

    Args:
        wd_graph (ig.Graph): instance of the parsed igraph
        answer_entities (list[str]): answer candidates
        question_entities (list[str]): question entities

    Returns:
        ig.Graph: extracted igraph subgraph
    """
    pathes = []
    for question_entity in question_entities:
        for answer_entity in answer_entities:
            pathes += wd_graph.get_shortest_paths(
                answer_entity[1:], to=question_entity[1:], mode="all"
            )
            pathes += wd_graph.get_shortest_paths(
                question_entity[1:], to=answer_entity[1:], mode="all"
            )

    neighbours_vertices = []
    vertices = [v for p in pathes for v in p]
    vertices = list(set(vertices + neighbours_vertices))
    subgraph = wd_graph.subgraph(vertices)

    subgraph.vs["name"] = ["Q" + str(e) for e in subgraph.vs["name"]]
    subgraph.es["name"] = ["P" + str(int(w)) for w in subgraph.es["weight"]]
    del subgraph.es["weight"]

    vertex_type = []
    for entity in subgraph.vs:
        if entity["name"] in answer_entities:
            curr_type = str(SubgraphNodeType.ANSWER_CANDIDATE_ENTITY)
        elif entity["name"] in question_entities:
            curr_type = str(SubgraphNodeType.QUESTIONS_ENTITY)
        elif entity in neighbours_vertices:
            curr_type = str(SubgraphNodeType.QUESTIONS_ENTITY_NEIGHBOUR)
        else:
            curr_type = str(SubgraphNodeType.INTERNAL)
        vertex_type.append(curr_type.rsplit(".", maxsplit=1)[-1])
    subgraph.vs["type"] = vertex_type

    return subgraph


def igraph_to_nx(subgraph: ig.Graph):
    """convert the igraph subgraph to networkx

    Args:
        subgraph (_type_): igraph subgraph

    Returns:
        nx.Graph: nx subgraph
    """
    nx_subgraph = subgraph.to_networkx(nx.DiGraph)
    for _, _, d_node in nx_subgraph.edges(data=True):
        d_node.pop("_igraph_index", None)
        d_node["name_"] = d_node["name"]
        d_node.pop("name", None)

    for node_id in nx_subgraph.nodes():
        nx_subgraph.nodes[node_id].pop("_igraph_index")
        nx_subgraph.nodes[node_id]["name_"] = nx_subgraph.nodes[node_id]["name"]
        nx_subgraph.nodes[node_id].pop("name", None)
    return nx_subgraph


def write_from_queue(save_jsonl_path: str, results_q: JoinableQueue):
    """given a queue, write the queue to the save_jsonl_path file

    Args:
        save_jsonl_path (str): path of the jsonl file
        results_q (JoinableQueue): result queue (to write our results from)
    """
    with open(save_jsonl_path, "a+", encoding="utf-8") as file_handler:
        while True:
            try:
                json_obj = results_q.get()
                file_handler.write(json_obj + "\n")
            except QueueEmpty:
                continue
            else:
                results_q.task_done()


def read_wd_graph(wd_graph_path: str) -> ig.Graph:
    """given the path, parse the triples and build
    igraph Wikidata KG

    Args:
        wd_graph_path (str): path of the txt triple file

    Returns:
        ig.Graph: parsed igraph of Wikidata KG
    """
    if wd_graph_path.split(".")[-1].lower() in ["pkl", "pickle"]:
        return ig.Graph.Read_Pickle(wd_graph_path)

    return ig.Graph.Read_Ncol(
        wd_graph_path,
        names=True,
        directed=True,
        weights=True,
    )


def find_subgraph_and_transform_to_json(
    wd_graph: ig.Graph, task_q: JoinableQueue, results_q: JoinableQueue
):
    """main function/proccess to extract subgraph and write to jsonl

    Args:
        wd_graph_path (str): wikidata kg with igraph
        task_q (JoinableQueue): task queue
        results_q (JoinableQueue): result queue
    """

    wd_graph.get_shortest_paths = memory.cache(wd_graph.get_shortest_paths)

    print(
        f"[{now()}]{proc_worker_header}[{os.getpid()}] Current process memory (Gb)",
        psutil.Process(os.getpid()).memory_info().rss / (1024.0**3),
    )
    while True:
        try:
            task_line = task_q.get()
            start_time = time.time()
            data = ujson.loads(task_line)
            try:
                subgraph = extract_subgraph(
                    wd_graph, data["answerEntity"], data["questionEntity"]
                )
            except ValueError as value_err:
                with open("ErrorsLog.jsonl", "a+", encoding="utf-8") as file:
                    data["error"] = str(value_err)
                    file.write(ujson.dumps(data) + "\n")
                    continue
            except Exception as general_exception:  # pylint: disable=broad-except
                print(str(general_exception))
                time.sleep(60)
                subgraph = extract_subgraph(
                    wd_graph, data["answerEntity"], data["questionEntity"]
                )

            nx_subgraph = igraph_to_nx(subgraph)
            data["graph"] = nx.node_link_data(nx_subgraph)

            results_q.put(ujson.dumps(data))
        except QueueEmpty:
            continue
        else:
            task_queue.task_done()
            print(
                f"[{now()}]{proc_worker_header}[{os.getpid()}] \
                SSP task completed ({time.time() - start_time}s)"
            )


if __name__ == "__main__":
    args = parse.parse_args()

    Path(args.save_jsonl_path).parents[0].mkdir(parents=True, exist_ok=True)

    # loading in our graph
    proc_worker_header = f"{BColors.OKGREEN}[Process Worker]{BColors.ENDC}"
    print(f"[{now()}]] Start loading WD Graph")
    parsed_wd_graph = read_wd_graph(args.igraph_wikidata_path)
    print(
        f"[{now()}]]{BColors.OKGREEN} \
            WD Graph loaded{BColors.ENDC}"
    )

    queue_max_size = int(args.n_jobs * 8)
    results_queue = JoinableQueue(maxsize=queue_max_size * 2)
    task_queue = JoinableQueue(maxsize=queue_max_size)
    writing_thread = Process(
        target=write_from_queue,
        args=[args.save_jsonl_path, results_queue],
        daemon=True,
    )
    writing_thread.start()

    processes = []
    for _ in range(args.n_jobs):
        p = Process(
            target=find_subgraph_and_transform_to_json,
            args=[parsed_wd_graph, task_queue, results_queue],
            daemon=True,
        )
        p.start()
        time.sleep(180)

    with open(
        args.subgraphs_dataset_prepared_entities_jsonl_path, "r", encoding="utf-8"
    ) as f:
        for idx, line in enumerate(f):
            if idx < args.skip_lines:
                continue

            task_queue.put(line)

            if idx % args.n_jobs == 0:
                print(
                    f"[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} {idx} tasks sent"
                )
                print(
                    f"[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} results_queue size: \
                        {results_queue.qsize():4d}; task_queue size: {task_queue.qsize():4d}"
                )

    print(f"[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} All tasks sent")
    task_queue.join()
    results_queue.join()
