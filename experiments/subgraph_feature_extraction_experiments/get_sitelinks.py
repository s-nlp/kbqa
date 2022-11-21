# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import json
from functools import lru_cache

@lru_cache(maxsize=None)
def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

@lru_cache(maxsize=None)
def run_sitelinks(entity_idx):

    try:
        endpoint_url = "https://query.wikidata.org/sparql"

        query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX schema: <http://schema.org/>
        SELECT (COUNT(DISTINCT ?sitelink) AS ?count) WHERE {

            ?item ?itemlabel wd:VALUE .
        
            ?sitelink schema:about ?item .
            ?sitelink schema:inLanguage "en" . 
            ?sitelink schema:isPartOf <https://commons.wikimedia.org/> .   

            }"""

        nquery = query.replace("VALUE", entity_idx)
        results = get_results(endpoint_url, nquery)

        for result in results["results"]["bindings"]:
            sitelinks = result

        return sitelinks["count"]["value"]
    except (SPARQLWrapper.SPARQLExceptions.EndPointInternalError, SPARQLWrapper.SPARQLExceptions.EndPointNotFound):
        return float('Nan')

@lru_cache(maxsize=None)
def run_query_in(entity_idx):

    url = "https://query.wikidata.org/sparql"

    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX wd: <http://www.wikidata.org/entity/> 

    SELECT (COUNT( DISTINCT * ) AS ?UniqueInlinks)  WHERE {

        ?item ?itemlabel wd:<ENTITY> .
 
        ?s ?p ?item .}""".replace(
        "<ENTITY>", entity_idx
    )

    try:
        request = requests.get(url, params={"format": "json", "query": query})
        data = request.json()
        return data["results"]["bindings"][0]["UniqueInlinks"]["value"]
    
    except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
        return float('NaN')

@lru_cache(maxsize=None)
def run_query_out(entity_idx):

    url = "https://query.wikidata.org/sparql"

    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX wd: <http://www.wikidata.org/entity/> 

    SELECT (COUNT( DISTINCT * ) AS ?UniqueOutlinks)  WHERE {

        ?item ?itemlabel wd:<ENTITY> .
 
        ?item ?p ?o .}""".replace(
        "<ENTITY>", entity_idx
    )

    try:
        request = requests.get(url, params={"format": "json", "query": query})
        data = request.json()
        return data["results"]["bindings"][0]["UniqueOutlinks"]["value"]

    except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
        return float('NaN')
