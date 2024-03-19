import requests
from joblib import Memory

memory = Memory("./cache", verbose=0)


@memory.cache
def get_wd_search_results(
    search_string: str,
    max_results: int = 500,
    language: str = "en",
    mediawiki_api_url: str = "https://www.wikidata.org/w/api.php",
    user_agent: str = None,
) -> list:
    params = {
        "action": "wbsearchentities",
        "language": language,
        "search": search_string,
        "format": "json",
        "limit": 50,
    }

    user_agent = "pywikidata" if user_agent is None else user_agent
    headers = {"User-Agent": user_agent}

    cont_count = 1
    results = []
    while cont_count > 0:
        params.update({"continue": 0 if cont_count == 1 else cont_count})

        reply = requests.get(mediawiki_api_url, params=params, headers=headers)
        reply.raise_for_status()
        search_results = reply.json()

        if search_results.get("success") != 1:
            raise Exception("WD search failed")
        else:
            for i in search_results["search"]:
                results.append(i["id"])

        if "search-continue" not in search_results:
            cont_count = 0
        else:
            cont_count = search_results["search-continue"]

        if cont_count > max_results:
            break

    return results
