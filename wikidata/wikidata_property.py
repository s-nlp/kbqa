import time
import requests
import logging
from wikidata.base import WikidataBase

class WikidataProperty(WikidataBase):
    """WikidataProperty - class for request properties between two entities with cache
    """
    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
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
        
        properties = [pdata['p']['value'] for pdata in properties_data]
        if return_id_only is True:
            properties = [self._wikidata_uri_to_id(prop) for prop in properties]

        return properties
        
    
    def _request_wikidata(self, entity1, entity2):
        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        select ?p where { 
            wd:<ENTITY1> ?p wd:<ENTITY2> .
        }
        """.replace("<ENTITY1>", entity1).replace("<ENTITY2>", entity2)

        def _try_request(query, url):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    headers={"Accept": "application/json"},
                )
                if request.status_code == 503: # Timeout
                    return None

                data = request.json()

                if len(data["results"]["bindings"]) == 0:
                    return None

                return data["results"]["bindings"]

            except requests.exceptions.ConnectionError as connection_exception:
                logging.error(str(connection_exception))
                raise connection_exception

            except ValueError:
                logging.info("sleep 60...")
                time.sleep(60)
                return self._try_request(query, url)

            except Exception:
                return None

        return _try_request(query, self.sparql_endpoint)

