import os
import pickle
from abc import ABC

from ..config import DEFAULT_CACHE_PATH


class CacheBase(ABC):
    """CacheBase - Abstract base class for storing something in cache file"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        cache_filename: str = "cache.pkl",
    ) -> None:
        self.cache_dir_path = cache_dir_path
        self.cache_filename = cache_filename
        self.cache = None

        self.cache_file_path = os.path.join(self.cache_dir_path, self.cache_filename)

        self.load_from_cache()

    def load_from_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "rb") as file:
                self.cache = pickle.load(file)

    def save_cache(self):
        if not os.path.exists(self.cache_dir_path):
            os.makedirs(self.cache_dir_path)

        with open(self.cache_file_path, "wb") as file:
            pickle.dump(self.cache, file)
