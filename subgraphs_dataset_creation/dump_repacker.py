# pylint: disable=c-extension-no-member
"""
script to parse the wikidata dump
"""
import argparse
import gzip
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
    default="latest-all.json.gz",
    type=str,
    help="choose the path for the wikidata dump graph (json.bz2)",
)

parser.add_argument(
    "--save_path",
    default="wikidata_dump_repack",
    type=str,
    help="choose the path to save the parse rdf triple txt of our graph",
)


def load_bz(data_path_bz2, lines_skip):
    """
    parse bz file, yield 1 record at a time
    """
    with gzip.open(data_path_bz2, mode="rt") as file:
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


def worker(save_dir, json_dump, idx, line, queue):  # pylint: disable=too-many-locals
    """
    worker parsing 1 record, retrieve the rdf triple form and
    put on the queue
    """
    try:
        # first or last line in file
        if line == "" or line == "\n" or line[0] in ("]", "["):
            return
        record = ujson.loads(line.rstrip(",\n"))

        if json_dump:
            with open(f"{save_dir}/res.json", "a+", encoding="utf-8") as file:
                ujson.dump(record, file, ensure_ascii=False, indent=4)

        rdf_triples = ""
        entity1 = pydash.get(record, "id")
        entity_label = pydash.get(record, "labels.en.value").replace(
            " ", "<space-replaced>"
        )
        lgl = f"\n#\t{entity1}"
        lgl += f"\n{entity_label}\t-1"

        # each key is the relationship between entity1 and 2
        for key in pydash.get(record, "claims").keys():
            for connected_ent_record in pydash.get(record, f"claims.{key}"):
                if (
                    pydash.get(connected_ent_record, "mainsnak.datavalue.type")
                    == "wikibase-entityid"
                ):
                    entity2 = pydash.get(
                        connected_ent_record, "mainsnak.datavalue.value.id"
                    )
                    # not including the Q and P
                    rdf_triples += f"{entity1[1:]}\t{entity2[1:]}\t{key[1:]}\n"
                    lgl += f"\n{entity2[1:]}\t{key[1:]}"

        msg = {
            "triples": rdf_triples,
            "labels": f"{entity1[1:]}\t{entity_label}\n",
            "lgl": lgl,
        }
        queue.put((idx, msg))

    except ujson.JSONDecodeError as error:
        print(f"Error idx: {idx}")
        print(error)
        print(line)


def init_worker():
    """in case we exit from parsing"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def listener(
    save_dir, queue, response_queue, lines_skipped
):  # pylint: disable=too-many-locals
    """
    listens for messages on our queue, write to the result file
    """
    output_triples = f"{save_dir}/wikidata_triples.txt"
    output_labels = f"{save_dir}/wikidata_triples_labels.txt"
    output_lgl = f"{save_dir}/wikidata_lgl.txt"
    checkpoint = f"{save_dir}/checkpoint.txt"
    last_written_entity = None
    with open(output_triples, "a+", encoding="utf-8") as file_triples:
        with open(output_labels, "a+", encoding="utf-8") as file_labels:
            with open(output_lgl, "a+", encoding="utf-8") as file_lgl:
                while True:
                    idx, msg = queue.get()
                    if msg == "kill":
                        response_queue.put(
                            f"KILL done: last written entity: {last_written_entity}"
                        )
                        break
                    if msg == "kp":
                        with open(checkpoint, "w", encoding="utf-8") as check_f:
                            check_f.write(
                                f"{str(idx + lines_skipped)}\n{last_written_entity}"
                            )
                        response_queue.put(
                            f"KP done: next part starts with idx: {idx + lines_skipped}, \
                                last written entity: {last_written_entity}"
                        )
                        continue
                    try:
                        space_index = msg["triples"].find("\t")
                        if space_index != -1:
                            last_written_entity = msg["triples"][:space_index]
                        file_triples.write(str(msg["triples"]))
                        file_triples.flush()
                        file_labels.write(str(msg["labels"]))
                        file_labels.flush()
                        file_lgl.write(str(msg["lgl"]))
                        file_lgl.flush()
                    except KeyboardInterrupt:
                        print("Exiting from parsing early while in writer!")
                        file_triples.flush()
                        file_labels.flush()
                        file_lgl.flush()
                        try:
                            sys.exit(130)
                        except SystemExit:
                            os._exit(130)  # pylint: disable=W0212


def main():  # pylint: disable=too-many-locals
    """main function"""
    args = parser.parse_args()

    checkpoint = f"{args.save_path}/checkpoint.txt"
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    number_of_processes = max(3, int(mp.cpu_count() * 2))
    # creating maner and queue
    manager = mp.Manager()
    task_queue = manager.Queue(number_of_processes * 4)
    listener_response_queue = manager.Queue(1)

    # for checkpoint
    lines_skipped = 0
    if Path(checkpoint).is_file() is True:
        with open(checkpoint, "r", encoding="utf-8") as checkpoint_f:
            lines_skipped = int(checkpoint_f.readline())

    print(f"Skip {lines_skipped} lines because of checkpoint")

    with mp.Pool(1, init_worker) as writing_pool:
        writing_pool.apply_async(
            listener,
            (args.save_path, task_queue, listener_response_queue, lines_skipped),
        )

        kp_amount = 10000
        with mp.Pool(number_of_processes, init_worker) as pool:
            try:
                # firing our workers to parse the record
                start = time.time()
                last_idx = None
                asyncs = []
                for idx, line in tqdm(load_bz(args.data_path, lines_skipped)):
                    if idx > 0 and idx % kp_amount == 0:
                        for res in asyncs:
                            res.wait()
                        asyncs = []
                        task_queue.put((idx, "kp"))
                        kp_description = listener_response_queue.get()
                        print(kp_description)

                    res = pool.apply_async(
                        worker,
                        (args.save_path, False, idx, line, task_queue),
                    )
                    asyncs.append(res)
                    last_idx = idx

                for res in asyncs:
                    res.wait()
                pool.close()
                task_queue.put((-1, "kill"))
                kill_description = listener_response_queue.get()
                print(kill_description)
                writing_pool.close()

                end = time.time()
                print(
                    f"time took {end - start}, total number of records: {last_idx + 1}"
                )
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
