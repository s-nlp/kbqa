[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Development Environment

### Build and run docker
```bash
docker build -f ./Dockerfile -t kbqa_dev ./
docker run -v $PWD:/workspace/kbqa/ --network host -ti kbqa_dev
```

### Prepare mGENRE

Download pretrained model and required files
```bash
# pretrained model
wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
tar -xvf fairseq_multilingual_entity_disambiguation.tar.gz

# data
wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
wget http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl
```

# Getting subgraphs dataset

Run `get_subgraphs_dataset.py`. The argument `--num_bad_candidates` is to indicate how many bad candidate answers are included in this dataset. The default value will be 5. 

```bash
python3 get_subgraphs_dataset.py
```

After running the above commands, the data will be in `/workspace/kbqa/subgraphs_dataset`. There will be 2 `.pkl` files. One will be with `subgraphs_edges_between.pkl`. There will be immediate edges between each shortest paths. On the other hand, the other will be `subgraphs_no_edges_between.pkl` with no immediate edges between the paths. 
### Wikidata utils
Wikidata SPARQL endpoint and Engine can be configured in `config.py`
By default, used query.wikidata.org

```python
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_ENGINE = "blazegraph"
```
# Getting subgraphs dataset

Run `get_subgraphs_dataset.py`. The argument `--num_bad_candidates` is to indicate how many bad candidate answers are included in this dataset. The default value will be 5. 

```bash
python3 get_subgraphs_dataset.py
```

After running the above commands, the data will be in `/workspace/kbqa/subgraphs_dataset`. There will be 2 `.pkl` files. One will be with `subgraphs_edges_between.pkl`. There will be immediate edges between each shortest paths. On the other hand, the other will be `subgraphs_no_edges_between.pkl` with no immediate edges between the paths. 
