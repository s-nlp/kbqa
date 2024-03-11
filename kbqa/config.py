from pathlib import Path
import os

LOG_FILENAME = "log.json"

SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES = [
    "facebook/bart-base",
    "facebook/bart-large",
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "google/t5-xl-ssm-nq",
    "google/t5-large-ssm",
    "google/t5-large-ssm-nq",
    "google/t5-small-ssm-nq",
    "google/flan-t5-small",
    "google/flan-t5-large",
    "s-nlp/t5_large_ssm_nq_mintaka",
]

SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT", "https://query.wikidata.org/sparql")
SPARQL_ENGINE = os.environ.get("SPARQL_ENGINE", "blazegraph")
# SPARQL_ENDPOINT = "http://localhost:7200/repositories/wikidata"
# SPARQL_ENGINE = "graphdb"

DEFAULT_CACHE_PATH = str(Path(__file__).parent / ".." / "cache_store")

DEFAULT_LRU_CACHE_MAXSIZE = os.environ.get("DEFAULT_LRU_CACHE_MAXSIZE", 16384)
