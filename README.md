[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
After we have parsed the Wikidata dump and have our Igraph triples representation, we are ready for subgraphs dataset generation. In order to do so, please run `./subgraphs_dataset_creation/generate_subgraphs_dataset.py`.

```bash
python3 get_subgraphs_dataset.py
```
The following are the available arguments:
 - `--num_bad_candidates` indicates how many bad candidate answers are included in this dataset. 
 - `number_of_pathes` indicates how many shortest paths we want to save in our subgraph. For each question entity to answer candidate, we will receive back multiple shortest paths of length $n$. 
 - `--model_result_path` indicates the path of mGENRE output (`results.csv`). 
 - `--edge_between_path` indicates whether or not the resulting subgraphs dataset will include immediate edges between each shortest paths. 
 - `--igraph_wikidata_path` indicates the path of where we parsed the above Wikidata json dump (`save_path` argument in the above section).
 - `--sqwd_jsonl_path` indicates the path of the parsed jsonl file (that provides all questions' entities and candidates in both Wikidata entity and label format).
 - `--save_dir` indicates the saving directory of our subgraphs dataset

 After running the above file, the data (`.pkl`) and the meta files (`.json`) will be in the file setup in `save_dir`.

#### Dataset formattings:


Each subgraphs and its meta file will be stored individually. The meta files will be stored in `meta_sgements` while the subgraphs files will be stored in `subgraphs_segments`. You can match each question to its meta file by the `id`. Each subgraphs are stored under its question folder. For instance, the subgraph file `question_0` will corresponds to `meta_id_0.json`. Inside of `save_dir`, we will have the subgraphs corresponding to this question (the number of subgraphs depends on the `num_bad_candidates` argument above). 

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
