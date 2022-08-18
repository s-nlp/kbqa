import time

import requests
from requests.exceptions import JSONDecodeError

from caches.base import CacheBase


class WikidataEntityToLabel(CacheBase):
    """WikidataEntityToLabel - Helper class for Wikidata Entities ID to labels"""

    def __init__(self, cache_dir_path: str = "./cache_store") -> None:
        super().__init__(cache_dir_path, "wikidata_entitiy_to_label.pkl")
        self.cache = {}
        self.load_from_cache()

    def get_entity_label(self, entity) -> str:
        entity = entity.upper()
        if entity not in self.cache:
            label = self._request_wikidata_label(entity)
            if label is not None:
                self.cache[entity] = label
                self.save_cache()
            else:
                return None
        return self.cache[entity]

    def _request_wikidata_label(self, entity_id: str) -> str:
        def _try_get_label():
            try:
                data = requests.get(
                    f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                )
                data = data.json()
                return data["entities"][entity_id]["labels"].get("en", {}).get("value")

            except JSONDecodeError:
                print("sleep 60...")
                time.sleep(60)
                return _try_get_label()

            except Exception:
                print(f"ERROR with request")
                print("sleep 60...")
                time.sleep(60)
                return _try_get_label()

        return _try_get_label()
