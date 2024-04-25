""" Parsing the jsonl reranking prediction file to gather reranking results (top@n)"""
from argparse import ArgumentParser, RawTextHelpFormatter
import json
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset
from pywikidata.utils import get_wd_search_results


DESCRIPTION = """Evaluation script for mintaka ranked predictions

Evaluate ranked predictions. If AnswerEntityID not provided and question is not count or YesNo type,
try to link AnswerString to Entity and compare with GT.
"""

EXAMPLE_OF_DATA_FORMAT = """
Example of data format in predictions_path:
    {
        "QuestionID": "ID1",
        "RankedAnswers": [
            {
                "AnswerEntityID": null,
                "AnswerString": "String of prediction",
                "Score": null
            },
            {
                "AnswerEntityID": "Q90",
                "AnswerString": "Paris",
                "Score": 0.99
            },
            ...
        ]
    },
"""

parser = ArgumentParser(
    description=DESCRIPTION,
    formatter_class=RawTextHelpFormatter,
)

# pylint: disable=line-too-long
parser.add_argument(
    "--predictions_path",
    help="Path to JSONL file with predictions" + EXAMPLE_OF_DATA_FORMAT,
    default="/workspace/storage/misc/subgraphs_reranking_runs/reranking_model_results/t5_xl_ssm/mpnet_text_only_reranking_seq2seq_xl_results.jsonl",
)

parser.add_argument(
    "--split",
    default="test",
    type=str,
    help="Mintaka dataset split.\ntest by default",
)


def label_to_entity(label: str, top_k: int = 1) -> list:
    """label_to_entity method to  linking label to WikiData entity ID
    by using elasticsearch Wikimedia public API
    Supported only English language (en)

    Parameters
    ----------
    label : str
        label of entity to search
    top_k : int, optional
        top K results from WikiData, by default 1

    Returns
    -------
    list[str] | None
        list of entity IDs or None if not found
    """
    try:
        elastic_results = get_wd_search_results(label, top_k, language="en")[:top_k]
    except:  # pylint: disable=bare-except
        elastic_results = []

    try:
        elastic_results.extend(
            get_wd_search_results(
                label.replace('"', "").replace("'", "").strip(), top_k, language="en"
            )[:top_k]
        )
    except:  # pylint: disable=bare-except
        return [None]

    if len(elastic_results) == 0:
        return [None]

    return list(dict.fromkeys(elastic_results).keys())[:top_k]


class EvalMintaka:
    """EvalMintaka Evaluation class for Mintaka ranked predictions"""

    def __init__(self):
        mintaka_ds = load_dataset("AmazonScience/mintaka")
        self.dataset = {
            "train": mintaka_ds["train"].to_pandas(),
            "validation": mintaka_ds["validation"].to_pandas(),
            "test": mintaka_ds["test"].to_pandas(),
        }

        # Extract Entities Names (Ids) from dataset records
        for _, df in self.dataset.items():
            df["answerEntityNames"] = df["answerEntity"].apply(
                self._get_list_of_entity_ids
            )

    def _get_list_of_entity_ids(self, answer_entities):
        return [e["name"] for e in answer_entities]

    def is_answer_correct(self, mintaka_record: pd.Series, answer: dict) -> bool:
        """to check whether an answer is correct or not

        Args:
            mintaka_record (pd.Series): row in the mintaka dataset
            answer (dict): answer dict; comprising of the answer entity and/or answer str

        Returns:
            bool: correct or not
        """
        if mintaka_record["complexityType"] in ["count", "yesno"]:
            return answer["AnswerString"] == mintaka_record["answerText"]
        else:
            if answer.get("AnswerEntityID") is None:
                answer["AnswerEntityID"] = label_to_entity(answer["AnswerString"])[0]

            if (
                answer.get("AnswerEntityID") is None
                and mintaka_record["answerText"] is not None
            ):
                return answer["AnswerString"] == mintaka_record["answerText"]

            return answer.get("AnswerEntityID") in mintaka_record["answerEntityNames"]

    def evaluate(self, predictions, split: str = "test", top_n: int = 10):
        """evaluate _summary_

        Parameters
        ----------
        predictions : List[Dict]
            Predictions in the following format:
            [
                {
                    "QuestionID": "ID1",
                    "RankedAnswers": [
                        {
                            "AnswerEntityID": None,
                            "AnswerString": "String of prediction",
                            "Score": None
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        _df = self.dataset[split]

        is_correct = []
        for prediction in tqdm(predictions, desc="Process predictions.."):
            question_idx = prediction["QuestionID"]
            mintaka_record = _df[_df["id"] == question_idx].iloc[0]
            is_answer_correct_results = []
            for _, answer in enumerate(prediction["RankedAnswers"]):
                is_answer_correct_results.append(
                    self.is_answer_correct(mintaka_record, answer)
                )

            is_correct.append(is_answer_correct_results)

        is_correct_df = pd.DataFrame(is_correct)
        is_correct_df["id"] = [p["QuestionID"] for p in predictions]
        is_correct_df = _df.merge(is_correct_df, on="id")

        if len(set(is_correct_df["id"]).symmetric_difference(_df["id"])) != 0:
            print(
                "WARNING: Not all questions have predictions, "
                "the results will be calculated only for the provided predictions "
                "without taking into account the unworthy ones."
            )

        # Format metrics based on is_correct matrix
        without_yesno_and_count_filter = is_correct_df["complexityType"].apply(
            lambda s: s not in ["yesno", "count"]
        )
        results = {
            "FULL Dataset": self._calculate_hits(is_correct_df, top_n),
            "Without Yes/No and Count": self._calculate_hits(
                is_correct_df[without_yesno_and_count_filter], top_n
            ),
        }
        for complexity_type in is_correct_df["complexityType"].unique():
            results[f"Only {complexity_type}"] = self._calculate_hits(
                is_correct_df[is_correct_df["complexityType"] == complexity_type],
                top_n,
            )
        return results

    def _calculate_hits(self, is_correct_df: pd.DataFrame, top_n: int = 10) -> dict:
        hits = {}
        for top in range(1, top_n + 1):
            hits[f"Hit@{top}"] = (
                is_correct_df[list(range(top))].apply(any, axis=1).mean()
            )
        return hits


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.predictions_path, "r", encoding="utf-8") as f:
        reranking_predictions = [json.loads(line) for line in f.readlines()]

    eval_mintaka = EvalMintaka()
    reranking_results = eval_mintaka.evaluate(reranking_predictions, args.split, 5)

    # save reranking results
    OUTPUT_DIR = "/".join(args.predictions_path.split("/")[:-1])
    output_path = f"{OUTPUT_DIR}/reranking_result.txt"
    with open(output_path, "w+", encoding="utf-8") as file_output:
        file_output.write("Hit scores: \n")
        for key, val in reranking_results.items():
            file_output.write(f"{key}")
            for hitkey, hitval in val.items():
                file_output.write(f"\t{hitkey:6} = {hitval:.6f}")
            file_output.write("\n")
