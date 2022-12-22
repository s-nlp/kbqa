from pathlib import Path

LOG_FILENAME = "log.json"

SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES = [
    "facebook/bart-base",
    "facebook/bart-large",
    "t5-small",
    "t5-base",
    "t5-large",
    "google/t5-large-ssm",
    "google/t5-large-ssm-nq",
    "google/t5-small-ssm-nq",
    "google/flan-t5-small",
    "google/flan-t5-large",
]

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_ENGINE = "blazegraph"

DEFAULT_CACHE_PATH = str(Path(__file__).parent / "cache_store")

DEFAULT_LRU_CACHE_MAXSIZE = 16384
