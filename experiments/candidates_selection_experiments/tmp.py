import pandas as pd
from joblib import Parallel, delayed
from kbqa.wikidata import Entity
from tqdm.auto import tqdm


train_df = pd.read_csv(
    "./wikidata-simplequestions/annotated_wd_data_train_answerable.txt",
    sep="\t",
    names=["S", "P", "O", "Q"],
)

valid_df = pd.read_csv(
    "./wikidata-simplequestions/annotated_wd_data_valid_answerable.txt",
    sep="\t",
    names=["S", "P", "O", "Q"],
)

def _proc(idx):
    try:
        e = Entity(idx)
        e.one_hop_neighbors
    except Exception as e:
        print(str(e))

Parallel(n_jobs=4)(
    delayed(_proc)(idx)
    for idx in tqdm(train_df['S'])
)

Parallel(n_jobs=4)(
    delayed(_proc)(idx)
    for idx in tqdm(valid_df['S'])
)