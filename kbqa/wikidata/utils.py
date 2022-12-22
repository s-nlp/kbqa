import time
from functools import lru_cache

import requests

from ..config import DEFAULT_LRU_CACHE_MAXSIZE, SPARQL_ENDPOINT
from ..logger import get_logger


logger = get_logger()


@lru_cache(maxsize=DEFAULT_LRU_CACHE_MAXSIZE)
def request_to_wikidata(query, sparql_endpoint=SPARQL_ENDPOINT):
    params = {"format": "json", "query": query}
    logger.info(
        {
            "msg": "Send request to Wikidata",
            "params": params,
            "endpoint": sparql_endpoint,
        }
    )
    response = requests.get(
        sparql_endpoint,
        params=params,
        headers={"Accept": "application/json"},
    )
    to_sleep = 0.2
    while response.status_code == 429:
        logger.warning(
            {
                "msg": "Request to wikidata endpoint failed",
                "params": params,
                "endpoint": sparql_endpoint,
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                },
            }
        )
        if "retry-after" in response.headers:
            to_sleep += int(response.headers["retry-after"])
        to_sleep += 0.5
        time.sleep(to_sleep)
        response = requests.get(
            sparql_endpoint,
            params=params,
            headers={"Accept": "application/json"},
        )
    return response.json()["results"]["bindings"]
