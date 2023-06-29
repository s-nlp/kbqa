import os
import json
import hashlib
import csv
import tempfile
import re
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

    def _create_tsv(self, file_path, entities1, entities2):
        """
        create a tsv file based on the current entity
        and its neighbor, save this tsv file


        Args:
            file_path (str): temporary path to save the tsv file
            entities1 (list[str]): entities to compare to
            entities2 (list[str]): entities we want to compare
        """
        with open(file_path.name, "w", encoding="utf8", newline="") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow(["q1", "q2"])
            for entity1, entity2 in zip(entities1, entities2):
                tsv_writer.writerow([entity1, entity2])

    def _chunks(self, lst, size):
        """Yield successive "size" chunks from lst."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def _validate_entity_id(self, entity_id):
        return re.fullmatch(r"[P|Q][0-9]+", entity_id) is not None

    def entity_number_id_to_qid(self, node):
        """check the validity of node (must be either ID (without Q) or Q{ID})

        Args:
            node (str or int): the node that we want to validate/verify

        Raises:
            ValueError: _description_
        """
        if isinstance(node, str) is not True:
            node = str(node)

        if self._validate_entity_id(node):
            return node
        node = "Q" + node

        if self._validate_entity_id(node):
            return node
        raise ValueError(
            f"Invalid node identifier provided, must be number of entity id or entity id, but {node} provided"
        )

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
            entities1 (list[str]): list of entities to compare to
            entities2 (list[str]): list of entities to compare with
            chunk_size (int): size of chunk (due to API limitation)

        Returns:
            pd.DataFrame: dataframe of comparison
        """
        result_dfs = []

        # we can only request up to 25 entities -> split into chunks
        entities1 = [self.entity_number_id_to_qid(node) for node in entities1]
        entities1_chunks = self._chunks(entities1, chunk_size)
        entities2 = [self.entity_number_id_to_qid(node) for node in entities2]
        entities2_chunks = self._chunks(entities2, chunk_size)

        # save final result df with hashkey
        pre_hash_result_key = "".join(entities1) + "".join(entities2)
        hash_result_key = hashlib.sha256(
            pre_hash_result_key.encode("utf-8")
        ).hexdigest()

        if hash_result_key in self.cache:  # if final result df exist -> return
            similarity_dict = self.cache[hash_result_key]
            return pd.DataFrame(similarity_dict)

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
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline=""
                ) as tmp_file:
                    self._create_tsv(tmp_file, entity1_chunk, entity2_chunk)

                    # create the post request
                    similarity_df = self._call_semantic_similarity(tmp_file.name, url)
                    self.cache[
                        hash_key
                    ] = similarity_df.to_dict()  # save final res our cache
                    self.save_cache()

            result_dfs.append(similarity_df)

        # remove emtpy similarities, combine to make final result df
        result_df = pd.concat(result_dfs)
        result_df.replace("", np.nan, inplace=True)
        result_df.dropna(inplace=True)

        if save_csv:  # save csv file to result path if True
            result_df.to_csv(
                f"{self.result_path}/{hash_result_key}.csv",
                index=False,
            )
        self.cache[
            hash_result_key
        ] = result_df.to_dict()  # save final result df to our cache
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
