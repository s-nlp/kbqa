from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase
from .utils import request_to_wikidata


class WikidataProperty(WikidataBase):
    """WikidataProperty - class for request properties between two entities with cache"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_properties.pkl", sparql_endpoint)
        self.cache = {}
        self.load_from_cache()

    def get_properties_between(self, entity1, entity2, return_id_only=False):
        key = (entity1, entity2)

        if key not in self.cache:
            properties_data = self._request_wikidata(entity1, entity2)
            if properties_data is not None:
                self.cache[key] = properties_data
                self.save_cache()
        else:
            properties_data = self.cache[key]

        properties = [pdata["p"]["value"] for pdata in properties_data]
        if return_id_only is True:
            properties = [self._wikidata_uri_to_id(prop) for prop in properties]

        return properties

    def _request_wikidata(self, entity1, entity2):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        select ?p where { 
            wd:<ENTITY1> ?p wd:<ENTITY2> .
        }
        """.replace(
            "<ENTITY1>", entity1
        ).replace(
            "<ENTITY2>", entity2
        )

        data = request_to_wikidata(query, self.sparql_endpoint)
        if len(data) == 0:
            return None

        return data
