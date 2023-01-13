from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pywikidata import Entity
from kbqa.candidate_selection import QuestionToRankInstanceOf
from seq2seq_dbs_answers_generation import load_params, load_datasets


if __name__ == "__main__":
    params, run_name = load_params()
    train_dataset, valid_dataset, test_dataset, question_col_name = load_datasets(
        params
    )
    datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    candidates_main_path = Path(
        f"/mnt/raid/data/kbqa/datasets/candidates/{run_name}/{params['seq2seq']['dataset']}/"
    )

    for split_name in ["valid", "test"]:
        candidates_df = pd.read_pickle(candidates_main_path / f"{split_name}.pkl")
        answer_cols = candidates_df.columns
        df = pd.concat([datasets[split_name].to_pandas(), candidates_df], axis=1)
        df["_index"] = df.index

        def l2e(l):
            try:
                return Entity.from_label(l)[0]
            except:
                pass

        df["seq2seq_results"] = df["answer_0"].apply(l2e)
        df["targets"] = df["object"].apply(lambda lst: [Entity(o) for o in lst])

        res = df.apply(lambda row: row["seq2seq_results"] in row["targets"], axis=1)

        res.sum() / res.index.size
