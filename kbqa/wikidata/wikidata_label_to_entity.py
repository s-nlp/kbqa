# pylint: disable=logging-format-interpolation
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=W0703
# pylint: disable=R1710
# pylint: disable=R0911
# pylint: disable=W1203

import time

import requests

from ..config import DEFAULT_CACHE_PATH
from .base import WikidataBase
from .wikidata_redirects import WikidataRedirectsCache
from ..logger import get_logger
from .utils import request_to_wikidata

logger = get_logger()


class WikidataLabelToEntity(WikidataBase):
    """WikidataEntityToLabel - class for request label of any wikidata entities with cahce"""

    def __init__(
        self,
        redirect_cache: WikidataRedirectsCache,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
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
                self.save_cache()
            else:
                self.cache[entity_name] = ""
                self.save_cache()

        return self.cache.get(entity_name)

    def __call__(self, label):
        return self.get_id(label)

    def _create_query(self, entity_name):
        query = """
        # PREFIX schema: <http://schema.org/>
        # PREFIX wikibase: <http://wikiba.se/ontology#>

        # SELECT ?item WHERE{
        #         ?item ?label "<ENTITY_NAME>"@en.
        #         ?article schema:about ?item .
        #         ?article schema:inLanguage "en" .
        # }
        SELECT * WHERE{
            ?item rdfs:label "<ENTITY_NAME>"@en .
        }
        """.replace(
            "<ENTITY_NAME>", entity_name
        )
        return query

    def _request_wikidata(self, entity_name):
        def run_request(entity_name, url):
            query = self._create_query(entity_name)
            # request = requests.get(
            #     url,
            #     params={"format": "json", "query": query},
            #     timeout=20,
            #     headers={"Accept": "application/json"},
            # )
            # data = request.json()
            data = request_to_wikidata(query, url)
            if len(data) > 0:
                return data[0]["item"]["value"].split("/")[-1]
            else:
                raise Exception("!!")

        def _try_request(entity_name, url, continue_redirecting=True):
            try:
                return run_request(entity_name, url)
            except ValueError:
                logger.info("sleep 2...")
                logger.error("false redirects")
                time.sleep(2)
                return ""
            except requests.exceptions.ConnectionError as connection_exception:
                logger.error(str(connection_exception))
                raise connection_exception
            except Exception:
                if self.redirect_cache is not None and continue_redirecting is True:
                    logger.error(
                        f'ERROR with entity "{entity_name}", fetching for redirects'
                    )

                    redirects = self.redirect_cache.get_redirects(entity_name)
                    if redirects == "No results found":
                        return ""
                    for redirect in redirects:
                        new_redirect = _try_request(
                            redirect, url, continue_redirecting=False
                        )
                        if new_redirect != "":
                            return new_redirect
                elif self.redirect_cache is None:
                    return None
                else:
                    logger.error(
                        f'ERROR with entity "{entity_name}", fetching for redirects, no redirects found (BREAK)'
                    )
                    return ""

        return _try_request(entity_name, self.sparql_endpoint)
