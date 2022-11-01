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

### Getting subgraphs dataset

#### Running to get the dataset
First of all, when running `get_subgraphs_dataset.py`, the command line arguments will be. 
- The argument `--num_bad_candidates` is to indicate how many bad candidate answers are included in this dataset. 
- The argument `--model_result_path` is to indicate the path of mGENRE output (`results.csv`). 
- The argument `--lang_title_wikidata_id_path` is to indicate the path of `lang_title2wikidataID-normalized_with_redirect.pkl`
- The argument `--marisa_trie_path` is to indicate the path of `titles_lang_all105_marisa_trie_with_redirect.pkl`
- The argument `--pretrained_mgenre_weight_path` is to indicate the path of the mGENRE pretrained weights - `fairseq_multilingual_entity_disambiguation`
- The argument `--batch_size` is to indicate the batch size of the questions of mGENRE output (`results.csv`). 
- The argument `--edge_between_path` is to indicate whether or not the resulting subgraphs dataset will include immediate edges between each shortest paths. 

```bash
python3 get_subgraphs_dataset.py
```

After running the above file, the data (`.pkl`) and the meta files (`.json`) will be in `/workspace/kbqa/subgraphs_dataset/dataset_v0`.

#### Dataset formattings:

Each subgraphs and its meta file will be stored individually. The meta files will be stored in `meta_sgements` while the subgraphs files will be stored in `subgraphs_segments`. You can match the subgraph to its meta file by the `id`. For instance, the subgraph file `graph_id_0.pkl` will corresponds to `meta_id_0.json`. 


### Wikidata utils
Wikidata SPARQL endpoint and Engine can be configured in `config.py`
By default, used query.wikidata.org

```python
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_ENGINE = "blazegraph"
```

For usage graphDB instance, write following config and do not forget to forward 7200 port to your machine  
```python
SPARQL_ENDPOINT = "http://localhost:7200/repositories/wikidata"
SPARQL_ENGINE = "graphdb"
```

