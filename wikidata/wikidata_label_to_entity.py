# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import time
import requests
import logging
from wikidata.base import WikidataBase
from wikidata.wikidata_redirects import WikidataRedirectsCache


class WikidataLabelToEntity(WikidataBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(
        self,
        redirect_cache: WikidataRedirectsCache,
        cache_dir_path: str = "./cache_store",
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
        }
        """.replace(
            "<ENTITY_NAME>", entity_name
        )
        return query

    def _try_request(self, query, url):
        try:
            request = requests.get(
                url,
                params={"format": "json", "query": query},
                timeout=20,
                headers={"Accept": "application/json"},
            )
            data = request.json()

            return data["results"]["bindings"][0]["item"]["value"].split("/")[-1]
        
        except requests.exceptions.ConnectionError as connection_exception:
            logging.error(str(connection_exception))
            raise connection_exception

        except ValueError:
            logging.info("sleep 60...")
            time.sleep(60)
            return self._try_request(query, url)

        except Exception:
            return None

    def _request_wikidata(self, entity_name):
        query = self._create_query(entity_name)
        res = self._try_request(query, self.sparql_endpoint)

        # if valid answer, return res
        if res is not None:
            return res

        logging.warning('ERROR with entity "{}", fetching for redirects'.format(entity_name))
        redirects = self.redirect_cache.get_redirects(entity_name)
        if redirects == "No results found":
            return ""

        for redirect in redirects:
            new_query = self._create_query(redirect)
            new_res = self._try_request(new_query, self.sparql_endpoint)

            # if we get a result for redirect, end loop and return
            if new_res is not None:
                return new_res

        # all redirects have been checked and no id
        return ""
