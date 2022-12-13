import requests
import time
import pickle
from functools import lru_cache

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


class LabelToEntity:
    def __init__(
        self,
    ):
        with open(
            "/bot/data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb"
        ) as f:
            self.lang_title_to_wikidata_id = pickle.load(f)

    @lru_cache(maxsize=16384)
    def text_to_id(self, x):
        set_of_ids = self.lang_title_to_wikidata_id.get(
            tuple(reversed(x.split(" >> ")))
        )
        if set_of_ids is None:
            return None
        else:
            return max(
                set_of_ids,
                key=lambda y: int(y[1:]),
            )
