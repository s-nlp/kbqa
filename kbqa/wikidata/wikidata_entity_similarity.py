import os
import json
import hashlib
import csv
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm

from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase


class WikidataEntitySimilarityCache(WikidataBase):
    """WikidataShortestPathCache - class for request shortest path from wikidata service
    with storing cache
    """

    def __init__(
        self,
        result_path: str,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
    ):
        super().__init__(cache_dir_path, "wikidata_entities_similarity.pkl")
        self.result_path = result_path
        self.cache = {}
        self.load_from_cache()

    def _create_tsv(self, file_path, entity_list, neighbors):
        """
        create a tsv file based on the current entity
        and its neighbor, save this tsv file


        Args:
            file_path (_type_): path to save the tsv file
            entity_list (_type_): the current entity (list of length neighbor)
            neighbors (_type_): list of neighbor to compare to the current entity
        """
        with open(file_path, "w", encoding="utf8", newline="") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow(["q1", "q2"])
            for entity1, entity2 in zip(entity_list, neighbors):
                tsv_writer.writerow([entity1, entity2])

    def _chunks(self, lst, size):
        """Yield successive "size" chunks from lst."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def get_entity_similarity(
        self,
        entities1,
        entities2,
        chunk_size=25,
        save_csv=False,
        url="https://kgtk.isi.edu/similarity_api",
    ):
        """
        given a list of multi hop entities, return list of
        sortesd entities based on similarity compare to our
        original entity

        Args:
            entities1 (str): list of entities to compare to
            entities2 (list:str): list of entities to compare with
            chunk_size (int): size of chunk (due to API limitation)

        Returns:
            pd.DataFrame: dataframe of comparison
        """
        result_dfs = []

        # we can only request up to 25 entities -> split into chunks
        entities1 = [f"Q{node}" for node in entities1]
        entities1_chunks = self._chunks(entities1, chunk_size)
        entities2 = [f"Q{node}" for node in entities2]
        entities2_chunks = self._chunks(entities2, chunk_size)

        for entity1_chunk, entity2_chunk in tqdm(
            zip(entities1_chunks, entities2_chunks)
        ):
            # key in our cache is SHA256(entity1 chunk + entity2 chunk)
            pre_hash_key = "".join(entity1_chunk) + "".join(entity2_chunk)
            hash_key = hashlib.sha256(pre_hash_key.encode("utf-8")).hexdigest()

            if hash_key in self.cache:  # check if key is already in cache
                # dict version of similarity_df is cached
                similarity_dict = self.cache[hash_key]
                similarity_df = pd.DataFrame(similarity_dict)
            else:
                # create the tsv with entity and its multi-hop
                file_path = f"{self.result_path}/{hash_key}.tsv"
                self._create_tsv(file_path, entity1_chunk, entity2_chunk)

                # create the post request
                url = "https://kgtk.isi.edu/similarity_api"
                similarity_df = self._call_semantic_similarity(file_path, url)
                self.cache[
                    hash_key
                ] = similarity_df.to_dict()  # save final res our cache
                self.save_cache()
                os.remove(file_path)  # remove the tsv file as we don't need it

            result_dfs.append(similarity_df)

        # remove emtpy similarities
        result_df = pd.concat(result_dfs)
        result_df.replace("", np.nan, inplace=True)
        result_df.dropna(inplace=True)

        # save result df with hashkey
        pre_hash_result_key = "".join(entities1) + "".join(entities2)
        hash_result_key = hashlib.sha256(
            pre_hash_result_key.encode("utf-8")
        ).hexdigest()
        if save_csv:  # save csv file to result path if True
            result_df.to_csv(
                f"{self.result_path}/{hash_result_key}.csv",
                index=False,
            )
        self.cache[hash_result_key] = result_df.to_dict()  # save our cache
        self.save_cache()

        return result_df

    def _call_semantic_similarity(self, input_path, url):
        """
        make the api request for similarity for the current tsv file

        Args:
            input_file (str): path of the tsv file
            url (_type_): api url

        Returns:
            pd.DataFrame: dataframe of comparison
        """
        file_name = os.path.basename(input_path)
        with open(input_path, mode="rb") as input_file:
            files = {"file": (file_name, input_file, "application/octet-stream")}
            resp = requests.post(url, files=files, params={"similarity_types": "all"})
            if resp.status_code == 200:  # if succesful
                resp_json = json.loads(resp.json())
                result_df = pd.DataFrame(resp_json)
            else:  # failure -> empty df
                result_df = pd.DataFrame()
        return result_df
