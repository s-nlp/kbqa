from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase
from .utils import request_to_wikidata


class WikidataEntityToLabel(WikidataBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_entity_to_label.pkl", sparql_endpoint
        )
        self.cache = {}
        self.load_from_cache()

    def get_label(self, entity_idx):
        if entity_idx is None:
            return None

        if entity_idx not in self.cache:
            label = WikidataEntityToLabel._request_wikidata(
                entity_idx, self.sparql_endpoint
            )
            if label is not None:
                self.cache[entity_idx] = label
                self.save_cache()

        return self.cache.get(entity_idx)

    def __call__(self, entity_idx):
        return self.get_label(entity_idx)

    @classmethod
    def _request_wikidata(cls, entity_idx, sparql_endpoint):
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX wd: <http://www.wikidata.org/entity/> 
        SELECT  *
        WHERE {
            wd:<ENTITY> rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
        } 
        """.replace(
            "<ENTITY>", entity_idx
        )
        data = request_to_wikidata(query, sparql_endpoint)
        if len(data) == 0:
            return None
        return data[0]["label"]["value"]
