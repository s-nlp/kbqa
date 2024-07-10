[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Knowledge Graph Question Answering

The main repository for projects related to KGQA techniques.

* `seq2seq.py` - main entry point to tune language models, evaluate and make predictions for KGQA datasets
* [subgraphs_dataset_creation](subgraphs_dataset_creation) - Includes scripts for mining subgraphs using iGraph.
* [experiments/subgraphs_reranking](experiments/subgraphs_reranking) - Experiments with subgraph ranking for the "Ranking Answers using a Large Language Model and Knowledge Graph" study.


# Development Environment

### Build and run docker
```bash
docker build -f ./Dockerfile -t kbqa_dev ./
docker run -v $PWD:/workspace/kbqa/ --network host -ti kbqa_dev
```

### Run Actions locally
For locally run actions (pylint & black check) [ACT](https://github.com/nektos/act) required. 
```bash
gh act -v
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

### Generating subgraphs dataset

 To generate our subgraph dataset, the desired code will be stored in `./subgraphs_dataset_creation`. This part consists of: 
 - parsing the Wikidata json dump to build our Wikidata graph via Igraph.
 - load our Igraph representation of Wikidata and generate the subgraph dataset

 #### Parsing Wikidata Json Dump 
 To parse the wikidata json dump, run the `./subgraphs_dataset_creation/parse_wikidata_dump.py`

 ```bash
 python3 get_subgraphs_dataset.py --data_path /workspace/storage/latest-all.json.bz2 --save_path /workspace/storage/wikidata_igraph
 ```
 where the arguments:
 - `data_path` refers to the path where the json dump is stored
 - `save_path` refers to the path where we want to save the igraph triple triples representation

 After running the above script, a `wikidata_triples.txt` file will be created within the saved path mentioned in the argument above. This triples text file is ready to be loaded via Igraph via:
 ```python
 # graph_path is where we stored wikidata_triples.txt
 igraph_wikidata = Graph.Read_Ncol(
             graph_path, names=True, directed=True, weights=True
         )
 ```
 Moreover, since parsing the json dump takes a decent amount of time, checkpoints were implemented. Let's say we are currently parsing the dump into `/workspace/storage/wikidata_igraph` (`--save_path` argument). For some unfortunate reason, our process crashed. You can simply rerun `./subgraphs_dataset_creation/parse_wikidata_dump.py` and it will automatically continue on where the crashed happen.

#### Running to get the dataset
After we have parsed the Wikidata dump and have our Igraph triples representation, we are ready for subgraphs dataset generation. In order to do so, please run `./subgraphs_dataset_creation/mining_subgraphs_dataset_processes.py`. This is also a time consuming process, thus we will utilize multi-processing.

```bash
python3 mining_subgraphs_dataset_processes.py
```
The following are the available arguments:
 - `--save_jsonl_path` indicates the path of the final resulting `jsonl` file (with our subgraphs)
 - `--igraph_wikidata_path` indicates the path of the file with our Igraph triples representation
 - `--subgraphs_dataset_prepared_entities_jsonl_path` indicates the path of the `jsonl` file representation of our `results.csv` (which include the answer candidates of the seq2seq run)
 - `--n_jobs` indicates how many jobs for our multi-processing scheme. **ATTENTION**: Each process require ~60-80Gb RAM.  
 - `--skip_lines` indicates the number of lines for skip in prepared_entities_jsonl file (from `--subgraphs_dataset_prepared_entities_jsonl_path`)

 After running the above file, the final data will be a `jsonl` file in the path `--save_jsonl_path`

#### Dataset formattings:

Each entry in the final `jsonl` file will represent one question-answer pair and its corresponding subgraph. One sample entry can be seen below:
```python
{"id":"fae46b21","question":"What man was a famous American author and also a steamboat pilot on the Mississippi River?","answerEntity":["Q893594"],"questionEntity":["Q1497","Q846570"],"groundTruthAnswerEntity":["Q7245"],"complexityType":"intersection","graph":{"directed":true,"multigraph":false,"graph":{},"nodes":[{"type":"INTERNAL","name_":"Q30","id":0},{"type":"QUESTIONS_ENTITY","name_":"Q1497","id":1},{"type":"QUESTIONS_ENTITY","name_":"Q846570","id":2},{"type":"ANSWER_CANDIDATE_ENTITY","name_":"Q893594","id":3}],"links":[{"name_":"P17","source":0,"target":0},{"name_":"P17","source":1,"target":0},{"name_":"P17","source":2,"target":0},{"name_":"P527","source":2,"target":3},{"name_":"P17","source":3,"target":0},{"name_":"P279","source":3,"target":2}]}}
```

### Pygraphviz

Install graphviz version 2.43.0 (0) manually
  
### Plots
run the script plot_subgraph_dataset_issue48.py in experiments folder to create dataset in "subgraph_plots" directory. The script auto distributes correct and wrong candidate graphs for the corresponding graph types and names the directory according to graph type.
- The argument `--json_graphs_directory` takes the directory path for subgraphs in json format.
- The argument `--meta_files_directory"` takes the directory path for meta files for corresponding subgraphs.
- The argument `--graph_type"` takes in a string as input, namely 'nx' or 'viz' for the networkx or graphviz format plots.

### CNN 
run cnn_classifier.py in reranking to get the confusion matrix and accuracy results with plots for lr, loss and accuracy.
- The argument `--batch_size` is for training batch size. 
- The argument `--train_epochs` is for number of training epochs.
- The argument `--directory` is for directory path of the plots (with "correct" and "wrong" sub directories).
- The argument `--test_samples` is for the number of samples to create the set. 

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

### Demo
For run Candidate Generation Demo webapp and TG bot
```bash
docker build -f ./Dockerfile.demo -t kbqa_bot ./
```

```bash
docker run \
    -p 7860:7860 \
    -v /home/salnikov/nlp2_seq2seq_runs/:/workspace/runs/ \
    --name kbqa_demo_bot \
    -e TG_QA_CG_BOT_KEY=<TG_BOT_KEY> \
    -e QA_CG_DEMO_MODEL_PATH=/workspace/runs/<PATH_TO_MODEL_DIR_INSIDE_DOCKER> \
    -e QA_CG_DEMO_MODEL_NAME=<MODEL_NAME> \
    -it kbqa_bot
```
