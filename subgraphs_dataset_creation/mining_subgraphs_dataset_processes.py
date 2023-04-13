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
    'save_jsonl_path',
    type=str,
    help='Path to JSONL file with resulting: data from subgraphs_dataset_prepared_entities_jsonl_path and graph',
)
parse.add_argument(
    '--igraph_wikidata_path',
    default='/mnt/storage/le/shortest_path/wikidata_igraph_v2/wikidata_triples.txt',
    type=str,
    help='Path to Ncol or Pkl file with WD whole graph',
)
parse.add_argument(
    '--subgraphs_dataset_prepared_entities_jsonl_path',
    default='/mnt/storage/le/shortest_path/sqwd_to_subgraphs_prepared_entities/sqwd_train.jsonl',
    type=str,
    help='Path to file with prepared entities with questions from dataset',
)
parse.add_argument(
    '--n_jobs',
    type=int,
    default=1,
    help='Number of parallel process to compute SSP. ATTENTION: Each process require ~60-80Gb RAM'
)
parse.add_argument(
    '--skip_lines',
    type=int,
    default=0,
    help='Number of lines for skip in prepared_entities_jsonl file',
)

memory = Memory('/tmp/mem', verbose=0)

def now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
    pathes = []
    for question_entity in question_entities:
        for answer_entity in answer_entities:
            pathes += wd_graph.get_shortest_paths(answer_entity[1:], to=question_entity[1:], mode="all")
            pathes += wd_graph.get_shortest_paths(question_entity[1:], to=answer_entity[1:], mode="all")

    # vertices = list(set([v for p in pathes for v in p]))
    neighbours_vertices = []
    # for question_entity in question_entities:
    #     neighbours_vertices += wd_graph.neighbors(
    #         wd_graph.vs.find(name=question_entity[1:]),
    #         mode='out',
    #     )
    
    vertices = [v for p in pathes for v in p]
    vertices = list(set(vertices + neighbours_vertices))
    subgraph = wd_graph.subgraph(vertices)

    subgraph.vs["name"] = ['Q'+str(e) for e in subgraph.vs["name"]]
    subgraph.es["name"] = ['P'+str(int(w)) for w in subgraph.es['weight']]
    del(subgraph.es['weight'])

    vertex_type = []
    for entity in subgraph.vs:
        if entity['name'] in answer_entities:
            vertex_type.append(str(SubgraphNodeType.ANSWER_CANDIDATE_ENTITY).split('.')[-1])
        elif entity['name'] in question_entities:
            vertex_type.append(str(SubgraphNodeType.QUESTIONS_ENTITY).split('.')[-1])
        elif entity in neighbours_vertices:
            vertex_type.append(str(SubgraphNodeType.QUESTIONS_ENTITY_NEIGHBOUR).split('.')[-1])
        else:
            vertex_type.append(str(SubgraphNodeType.INTERNAL).split('.')[-1])

    subgraph.vs['type'] = vertex_type

    return subgraph


def igraph_to_nx(subgraph):
    nx_subgraph = subgraph.to_networkx(nx.DiGraph)
    for _, _, d in nx_subgraph.edges(data=True):
        d.pop("_igraph_index", None)
        d['name_'] = d['name']
        d.pop('name', None)

    for id in nx_subgraph.nodes():
        nx_subgraph.nodes[id].pop('_igraph_index')
        nx_subgraph.nodes[id]['name_'] = nx_subgraph.nodes[id]['name']
        nx_subgraph.nodes[id].pop('name', None)
    return nx_subgraph



def write_from_queue(save_jsonl_path: str, results_queue: JoinableQueue):
    with open(save_jsonl_path, 'a+') as file_handler:
        while True:
            try:
                json_obj = results_queue.get()
                file_handler.write(json_obj + '\n')
            except QueueEmpty:
                continue
            else:
                results_queue.task_done()


def read_wd_graph(wd_graph_path: str) -> ig.Graph:
    if wd_graph_path.split('.')[-1].lower() in ['pkl', 'pickle']:
        return ig.Graph.Read_Pickle(wd_graph_path)
    else:
        return ig.Graph.Read_Ncol(
            wd_graph_path,
            names=True,
            directed=True,
            weights=True,
        )

def find_subgraph_and_transform_to_json(wd_graph_path, task_queue, results_queue):
    proc_worker_header = f"{BColors.OKGREEN}[Process Worker]{BColors.ENDC}"
    print(f'[{now()}]{proc_worker_header}[{os.getpid()}] Start loading WD Graph')
    print(f"[{now()}]{proc_worker_header}[{os.getpid()}] Current process memory (Gb)", psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3))
    wd_graph = read_wd_graph(wd_graph_path)
    wd_graph.get_shortest_paths = memory.cache(wd_graph.get_shortest_paths)
    print(f'[{now()}]{proc_worker_header}[{os.getpid()}]{BColors.OKGREEN} WD Graph loaded{BColors.ENDC}')
    print(f"[{now()}]{proc_worker_header}[{os.getpid()}] Current process memory (Gb)", psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3))
    while True:
        try:
            task_line = task_queue.get()
            start_time = time.time()
            data = ujson.loads(task_line)
            try:
                subgraph = extract_subgraph(wd_graph, data['answerEntity'], data['questionEntity'])
            except ValueError as e:
                with open('ErrorsLog.jsonl', 'a+') as f:
                    data['error'] = str(e)
                    f.write(ujson.dumps(data)+'\n')
                    continue
            except Exception as e:
                print(str(e))
                time.sleep(60)
                subgraph = extract_subgraph(wd_graph, data['answerEntity'], data['questionEntity'])

            nx_subgraph = igraph_to_nx(subgraph)
            data['graph'] = nx.node_link_data(nx_subgraph)

            results_queue.put(ujson.dumps(data))
        except QueueEmpty:
            continue
        else:
            task_queue.task_done()
            print(f'[{now()}]{proc_worker_header}[{os.getpid()}] SSP task complite ({time.time() - start_time}s)')


if __name__ == '__main__':
    args = parse.parse_args()
    print(args)

    Path(args.save_jsonl_path).parents[0].mkdir(parents=True, exist_ok=True)

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
            args=[args.igraph_wikidata_path, task_queue, results_queue],
            daemon=True,
        )
        p.start()
        time.sleep(180)

    with open(args.subgraphs_dataset_prepared_entities_jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx < args.skip_lines:
                continue

            task_queue.put(line)

            if idx % args.n_jobs == 0:
                print(f"[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} {idx} tasks sent")
                print(f"[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} results_queue size: {results_queue.qsize():4d}; task_queue size: {task_queue.qsize():4d}")

    print(f'[{now()}]{BColors.HEADER}[Main Thread]{BColors.ENDC} All tasks sent')
    task_queue.join()
    results_queue.join()
