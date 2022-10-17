[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Development

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

### Subgraph Plotting
Run the script plot_subgraph_dataset_issue59.py on the subgraph_edges_between.pkl or subgraph_no_edges_between.pkl files to plot subgraphs and auto create subgraph_plots directory with "correct" and "wrong" sub directories for gold standard and seq2seq generated candidates respectively (not necessarily unique) 

### Running CNN 
Place the generated subgraph plot dataset into "training" and "validation" sub directories inside subgraph_plots appropriately and run the script cnn_classifier with flag --test_samples for the number test plots to measure the metrics. In addition the script takes additional flags --batch_size, --train_epochs, --directory for batch size, number of training epochs and subgraph plot directory address respectively. Post training the plots will be saved in the working directory and the results will be outputted  

### Wikidata utils
Wikidata SPARQL endpoint and Engine can be configured in `config.py`
By default, used query.wikidata.org

```python
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_ENGINE = "blazegraph"
```
