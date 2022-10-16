from caches.base import CacheBase
import os
from urllib.parse import urlparse
from config import SPARQL_ENDPOINT


class WikidataBase(CacheBase):
    """
    WikidataBase - Abstract base class for working with Wikidata SPARQL endpoints
    and storing results in cache file
    """

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
        cache_filename: str = "cache.pkl",
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(cache_dir_path, cache_filename)

        self.sparql_endpoint = sparql_endpoint
        if self.sparql_endpoint is None:
            self.sparql_endpoint = SPARQL_ENDPOINT

        parsed_sparql_endpoint = urlparse(self.sparql_endpoint)
        path_from_sparql_endpoint = (
            parsed_sparql_endpoint.netloc + "|" + parsed_sparql_endpoint.path[1:]
        ).replace("/", "_")
        self.cache_dir_path = os.path.join(
            self.cache_dir_path, path_from_sparql_endpoint
        )
        self.cache_file_path = os.path.join(
            self.cache_dir_path,
            self.cache_filename,
        )

        self.cache = {}
        self.load_from_cache()
    
    def _wikidata_uri_to_id(self, uri):
        return uri.split("/")[-1].split("-")[0]
