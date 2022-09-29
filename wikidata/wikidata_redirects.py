from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List
from wikidata.base import WikidataBase


class WikidataRedirectsCache(WikidataBase):
    """WikidataRedirectsCache - Helper class for Wikidata Redirects
    request redirects from wikidata and store results to cache
    """

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
        sparql_endpoint: str = "http://dbpedia.org/sparql",
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_redirects.pkl", sparql_endpoint)
        self.cache = {}

    def get_redirects(self, term: str) -> List[str]:
        nterm = self._term_preprocess(term)

        if nterm not in self.cache:
            redirects = self._request_dbpedia(nterm)
            if "Problem communicating with the server" in redirects[0]:
                return redirects
            else:
                self.cache[nterm] = redirects
                self.save_cache()

        return self.cache[nterm]

    def _term_preprocess(self, term: str) -> str:
        term = term.strip()
        return term.capitalize().replace(" ", "_")

    def _request_dbpedia(self, nterm: str) -> List[str]:
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE 
        { 
        {
        <http://dbpedia.org/resource/VALUE> <http://dbpedia.org/ontology/wikiPageRedirects> ?x.
        ?x rdfs:label ?label.
        }
        UNION
        { 
        <http://dbpedia.org/resource/VALUE> <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
        ?x <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
        ?x rdfs:label ?label.
        }
        UNION
        {
        ?x <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/VALUE>.
        ?x rdfs:label ?label.
        }
        UNION
        { 
        ?y <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/VALUE>.
        ?x <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
        ?x rdfs:label ?label.
        }
        FILTER (lang(?label) = 'en')
        }
        """

        nquery = query.replace("VALUE", nterm)

        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(nquery)
        rterms = []
        sparql.setReturnFormat(JSON)
        try:
            ret = sparql.query()
            results = ret.convert()
            is_request_good = True
        except Exception:
            results = ""
            is_request_good = False

        if is_request_good is False:
            return "Problem communicating with the server: ", results
        elif len(results["results"]["bindings"]) == 0:
            return "No results found"
        else:
            for result in results["results"]["bindings"]:
                label = result["label"]["value"]
                rterms.append(label)

        return rterms
