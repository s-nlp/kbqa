# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import time
import requests
from wikidata.base import WikidataBase
from wikidata.wikidata_redirects import WikidataRedirectsCache


class WikidataLabelToEntity(WikidataBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(
        self,
        redirect_cache: WikidataRedirectsCache,
        cache_dir_path: str = "./cache_store_entity_id",
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_entity_to_id.pkl", sparql_endpoint)
        self.cache = {}
        self.load_from_cache()
        self.redirect_cache = redirect_cache

    def get_id(self, entity_name):
        if entity_name not in self.cache:
            entity_id = self._request_wikidata(entity_name)
            if entity_id is not None:
                self.cache[entity_name] = entity_id

        return self.cache.get(entity_name)

    def _create_query(self, entity_name):
        query = """
        PREFIX schema: <http://schema.org/>
        PREFIX wikibase: <http://wikiba.se/ontology#>

        SELECT ?item WHERE{
                ?item ?label "<ENTITY_NAME>"@en.
                ?article schema:about ?item .
                ?article schema:inLanguage "en" .
                ?article schema:isPartOf <https://en.wikipedia.org/>
        }
        """.replace(
            "<ENTITY_NAME>", entity_name
        )
        return query

    def _request_wikidata(self, entity_name):
        query = self._create_query(entity_name)

        def _try_request(query, url):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    timeout=20,
                    headers={"Accept": "application/json"},
                )
                data = request.json()

                return data["results"]["bindings"][0]["item"]["value"].split("/")[-1]

            except ValueError:
                print("sleep 60...")
                time.sleep(60)
                return _try_request(query, url)

            except Exception as general_exception:
                print(
                    'ERROR with entity "{}", fetching for redirects'.format(entity_name)
                )
                redirects = self.redirect_cache.get_redirects(entity_name)

                if redirects == "No results found":
                    raise Exception(
                        "NO ENTITY FOUND FOR THE CURRENT LABEL"
                    ) from general_exception
                for redirect in redirects:
                    new_query = self._create_query(redirect)
                    return _try_request(new_query, url)

        return _try_request(query, self.sparql_endpoint)
