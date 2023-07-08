# pylint: disable=c-extension-no-member
"""
script to parse the wikidata dump
"""
import argparse
import bz2
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
from pathlib import Path

import itertools
import pydash
import ujson
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="/workspace/storage/latest-all.json.bz2",
    type=str,
    help="choose the path for the wikidata dump graph (json.bz2)",
)

parser.add_argument(
    "--save_path",
    default="/workspace/storage/wikidata_igraph_v3_test",
    type=str,
    help="choose the path to save the parse rdf triple txt of our graph",
)


def load_bz(data_path_bz2, lines_skip):
    """
    parse bz file, yield 1 record at a time
    """
    with bz2.open(data_path_bz2, mode="rt") as file:
        file.read(2)  # skip first two bytes: "{\n"
        for idx, line in enumerate(itertools.islice(file, lines_skip, None)):
            yield idx, line


def save_pickle(pkl_file, dicc):
    """save a pkl file"""
    with open(pkl_file, "wb+") as file:
        pickle.dump(dicc, file)


def open_pickle(pkl_file):
    """open a pkl file"""
    with open(pkl_file, "rb") as file:
        res = pickle.load(file)
    return res


def worker(save_dir, json_dump, idx, line, queue):
    """
    worker parsing 1 record, retrieve the rdf triple form and
    put on the queue
    """
    try:
        record = ujson.loads(line.rstrip(",\n"))

        if json_dump:
            with open(f"{save_dir}/res.json", "a+", encoding="utf-8") as file:
                ujson.dump(record, file, ensure_ascii=False, indent=4)

        rdf_triples = ""
        entity1 = pydash.get(record, "id")

        # each key is the relationship between entity1 and 2
        for key in pydash.get(record, "claims").keys():
            for connected_entity_record in pydash.get(record, f"claims.{key}"):
                if pydash.has(connected_entity_record, "mainsnak.datavalue.type"):
                    if (
                        pydash.get(connected_entity_record, "mainsnak.datavalue.type")
                        == "wikibase-entityid"
                    ):
                        entity2 = pydash.get(
                            connected_entity_record, "mainsnak.datavalue.value.id"
                        )
                        # not including the Q and P
                        rdf_triples += f"{entity1[1:]}\t{entity2[1:]}\t{key[1:]}\n"
        queue.put((idx, rdf_triples))

    except ujson.JSONDecodeError:
        pass


def init_worker():
    """in case we exit from parsing"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def listener(save_dir, queue, lines_skipped):
    """
    listens for messages on our queue, write to the result file
    """
    output = f"{save_dir}/wikidata_triples.txt"
    checkpoint = f"{save_dir}/checkpoint.txt"
    with open(output, "a+", encoding="utf-8") as file:
        while True:
            idx, msg = queue.get()
            if msg == "kill":
                break
            try:
                file.write(str(msg))
                file.flush()
            except KeyboardInterrupt:
                print("Exiting from parsing early while in writer!")
                file.flush()
                try:
                    sys.exit(130)
                except SystemExit:
                    os._exit(130)  # pylint: disable=W0212

            if idx > 0 and idx % 1000 == 0:
                with open(checkpoint, "w+", encoding="utf-8") as check_f:
                    check_f.write(str(idx + lines_skipped))


def main():
    """main function"""
    args = parser.parse_args()
    checkpoint = f"{args.save_path}/checkpoint.txt"
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    number_of_processes = max(3, int(mp.cpu_count() * 2))
    # creating maner and queue
    manager = mp.Manager()
    task_queue = manager.Queue(number_of_processes * 4)

    # for checkpoint
    lines_skipped = 0
    if Path(checkpoint).is_file() is True:
        with open(checkpoint, "r", encoding="utf-8") as checkpoint_f:
            lines_skipped = int(checkpoint_f.readline())

    print(f"Skip {lines_skipped} lines because of checkpoint")

    with mp.Pool(1, init_worker) as writing_pool:
        writing_pool.apply_async(listener, (args.save_path, task_queue, lines_skipped))

        with mp.Pool(number_of_processes, init_worker) as pool:
            try:
                # firing our workers to parse the record
                start = time.time()
                last_idx = None
                for idx, line in tqdm(load_bz(args.data_path, lines_skipped)):
                    pool.apply_async(
                        worker,
                        (args.save_path, False, idx, line, task_queue),
                    )

                end = time.time()
                print(f"time took {end-start}, total number of records: {last_idx}")

                # now we are done, kill the listener
                task_queue.put((-1, "kill"))

                while not task_queue.empty():
                    pass
            except KeyboardInterrupt:
                print("Exiting from main pool early!")
                # terminate and joining since we are interrupting
                pool.terminate()
                writing_pool.terminate()
                pool.join()
                writing_pool.join()
    # everything parsed correctly, joining our workers
    pool.join()
    writing_pool.join()


if __name__ == "__main__":
    main()
