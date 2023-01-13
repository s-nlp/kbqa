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

    _run_name = "_".join(run_name.split("_")[:2])
    linked_entities_main_path = Path(
        f"/mnt/raid/data/kbqa/datasets/linked_entities/{_run_name}/{params['seq2seq']['dataset']}/"
    )

    for split_name in [
        "test",
    ]:
        candidates_df = pd.read_pickle(candidates_main_path / f"{split_name}.pkl")
        answer_cols = candidates_df.columns
        linked_entities_df = pd.read_pickle(
            linked_entities_main_path / f"{split_name}.pkl"
        )
        df = pd.concat(
            [
                datasets[split_name].to_pandas(),
                linked_entities_df[["selected_entities"]],
                candidates_df,
            ],
            axis=1,
        )
        df["_index"] = df.index

        for only_forward_one_hop in [
            True,
        ]:

            def _row_processing(row):
                try:
                    answers_candidates = []
                    for lbl in row[answer_cols].dropna().unique():
                        try:
                            answers_candidates.extend(Entity.from_label(lbl)[:1])
                        except ValueError:
                            pass
                    question_entities = [Entity(e) for e in row["selected_entities"]]

                    qtr = QuestionToRankInstanceOf(
                        row[question_col_name],
                        question_entities,
                        answers_candidates,
                        only_forward_one_hop=only_forward_one_hop,
                    )

                    results = qtr.final_answers()
                    results_df = pd.DataFrame(
                        results,
                        columns=[
                            "property",
                            "entity",
                            "instance_of_score",
                            "forward_one_hop_neighbors_score",
                            "answers_candidates_score",
                            "property_question_intersection_score",
                        ],
                    )
                    results_df["property"] = results_df["property"].apply(
                        lambda p: p.idx if p is not None else None
                    )
                    results_df["entity"] = results_df["entity"].apply(
                        lambda p: p.idx if p is not None else None
                    )

                    Path(
                        f"/mnt/raid/data/kbqa/datasets/selected_candidates/{run_name}/{params['seq2seq']['dataset']}/{split_name}/fw_only_{only_forward_one_hop}/"
                    ).mkdir(parents=True, exist_ok=True)
                    results_df.to_pickle(
                        f"/mnt/raid/data/kbqa/datasets/selected_candidates/{run_name}/{params['seq2seq']['dataset']}/{split_name}/fw_only_{only_forward_one_hop}/{row['_index']}.pkl"
                    )
                    results_df.to_json(
                        f"/mnt/raid/data/kbqa/datasets/selected_candidates/{run_name}/{params['seq2seq']['dataset']}/{split_name}/fw_only_{only_forward_one_hop}/{row['_index']}.json"
                    )
                    # print(f"Dumped to /mnt/raid/data/kbqa/datasets/selected_candidates/{run_name}/{params['seq2seq']['dataset']}/{split_name}/fw_only_{only_forward_one_hop}/")
                except Exception as e:
                    print(row["_index"], str(e))

            Parallel(n_jobs=6)(
                delayed(_row_processing)(row)
                for _, row in tqdm(
                    df.iterrows(),
                    total=df.index.size,
                    desc=f"split_name:{split_name}; only_forward_one_hop:{only_forward_one_hop}",
                )
            )
