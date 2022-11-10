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
<<<<<<< HEAD
### plots
run the script plot_subgraph_dataset.py in experiments folder to create dataset in "subgraph_plots" directory. It will automatically create subdirectories named "correct" and "wrong" for gold standard and seq2seq predicted sequences respectively 

### CNN 
run cnn_classifier in reranking to get the confusion matrix and accuracy results with plots for lr, loss and accuracy
=======

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

After running the above file, the data (`.pkl`) will be in `/workspace/kbqa/subgraphs_dataset`.

#### Dataset formattings:

The subgraphs are formatted sequentially based on the questions on `/workspace/kbqa/subgraphs_dataset/results.csv`. The good candidate subgraph will be first; then the bad candidates will follow. 

For example, let's say the `num_bad_candidates = 3`, the subgraphs array will look like:

```python
subgraphs_arr = [question1_good_candidate, question1_bad_candidate1, question1_bad_candidate2, question1_bad_candidate3, question2_good_candidate, question2_bad_candidate1, question2_bad_candidate2, question2_bad_candidate3, ...]
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
>>>>>>> master

For usage graphDB instance, write following config and do not forget to forward 7200 port to your machine  
```python
SPARQL_ENDPOINT = "http://localhost:7200/repositories/wikidata"
SPARQL_ENGINE = "graphdb"
```

