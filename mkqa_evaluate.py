""" Parsing the jsonl reranking prediction file to gather reranking results (top@n)"""
from argparse import ArgumentParser, RawTextHelpFormatter
import ast
import json
import os
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset
from pywikidata.utils import get_wd_search_results


DESCRIPTION = """Evaluation script for MKQA ranked predictions

Evaluate ranked predictions. If AnswerEntityID not provided,
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
    default="/workspace/storage/misc/subgraphs_reranking_runs/reranking_model_results/t5_large_ssm/mpnet_highlighted_t5_sequence_reranking_seq2seq_large_2_results.jsonl",
)

parser.add_argument(
    "--split",
    default="test",
    type=str,
    help="MKQA dataset split.\ntest by default",
)

parser.add_argument(
    "--force",
    action="store_true",
    help="Force evaluation even if output file already exists",
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


class EvalMKQA:
    """EvalMKQA Evaluation class for MKQA ranked predictions"""

    def __init__(self):
        mkqa_ds = load_dataset("Dms12/mkqa_mintaka_format_with_question_entities")
        self.dataset = {
            "train": mkqa_ds["train"].to_pandas(),
            "validation": mkqa_ds["validation"].to_pandas(),
            "test": mkqa_ds["test"].to_pandas(),
        }

        # Extract Entities Names (Ids) from dataset records
        for _, df in self.dataset.items():
            df["answerEntityNames"] = df["answerEntity"].apply(
                self._get_list_of_entity_ids
            )

    def _get_list_of_entity_ids(self, answer_entities):
        return [e["name"] for e in answer_entities]

    def is_answer_correct(self, mkqa_record: pd.Series, answer: dict) -> bool:
        """to check whether an answer is correct or not

        Args:
            mkqa_record (pd.Series): row in the MKQA dataset
            answer (dict): answer dict; comprising of the answer entity and/or answer str

        Returns:
            bool: correct or not
        """
        answer_entity_id = answer.get("AnswerEntityID")
        
        # Parse AnswerEntityID if it's a string representation of a list
        if answer_entity_id is not None and isinstance(answer_entity_id, str):
            if answer_entity_id.startswith("[") and answer_entity_id.endswith("]"):
                try:
                    parsed = ast.literal_eval(answer_entity_id)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        answer_entity_id = parsed[0]
                    else:
                        answer_entity_id = None
                except (ValueError, SyntaxError):
                    answer_entity_id = None
        
        if answer_entity_id is None:
            if answer.get("AnswerString") is not None:
                answer_entity_id = label_to_entity(answer["AnswerString"])[0]
            else:
                answer_entity_id = None

        if (
            answer_entity_id is None
            and mkqa_record["answerText"] is not None
            and answer.get("AnswerString") is not None
        ):
            return answer["AnswerString"] == mkqa_record["answerText"]

        if answer_entity_id is None:
            return False

        return answer_entity_id in mkqa_record["answerEntityNames"]

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

        import concurrent.futures

        def process_prediction(prediction):
            question_idx = int(prediction["QuestionID"])
            matching_records = _df[_df["id"] == question_idx]
            if len(matching_records) == 0:
                raise ValueError(f"QuestionID {question_idx} not found in dataset")
            mkqa_record = matching_records.iloc[0]
            is_answer_correct_results = [
                self.is_answer_correct(mkqa_record, answer)
                for answer in prediction["RankedAnswers"]
            ]
            return is_answer_correct_results

        is_correct = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(
                tqdm(
                    executor.map(process_prediction, predictions),
                    total=len(predictions),
                    desc="Process predictions.."
                )
            )
            is_correct.extend(results)

        is_correct_df = pd.DataFrame(is_correct)
        is_correct_df["id"] = [int(p["QuestionID"]) for p in predictions]
        is_correct_df = _df.merge(is_correct_df, on="id")

        if len(set(is_correct_df["id"]).symmetric_difference(_df["id"])) != 0:
            print(
                "WARNING: Not all questions have predictions, "
                "the results will be calculated only for the provided predictions "
                "without taking into account the unworthy ones."
            )

        # Format metrics based on is_correct matrix
        results = {
            "FULL Dataset": self._calculate_hits(is_correct_df, top_n),
        }
        return results

    def _calculate_hits(self, is_correct_df: pd.DataFrame, top_n: int = 10) -> dict:
        hits = {}
        numeric_cols = [col for col in is_correct_df.columns if isinstance(col, int)]
        for top in range(1, top_n + 1):
            cols_to_use = [col for col in range(top) if col in numeric_cols]
            if not cols_to_use:
                hits[f"Hit@{top}"] = 0.0
            else:
                hits[f"Hit@{top}"] = (
                    is_correct_df[cols_to_use].apply(any, axis=1).mean()
                )
        return hits


if __name__ == "__main__":
    args = parser.parse_args()

    OUTPUT_DIR = "/".join(args.predictions_path.split("/")[:-1])
    run_name = args.predictions_path.split("/")[-1]
    output_path = f"{OUTPUT_DIR}/reranking_result_{run_name}.txt"

    if os.path.exists(output_path) and not args.force:
        print(f"Output file already exists: {output_path}")
        print("Skipping evaluation. Use --force to re-evaluate.")
    else:
        with open(args.predictions_path, "r", encoding="utf-8") as f:
            reranking_predictions = [json.loads(line) for line in f.readlines()]

        eval_mkqa = EvalMKQA()
        reranking_results = eval_mkqa.evaluate(reranking_predictions, args.split, 5)

        with open(output_path, "w+", encoding="utf-8") as file_output:
            file_output.write("Hit scores:\n")
            for key, val in reranking_results.items():
                file_output.write(f"{key}")
                for hitkey, hitval in sorted(val.items(), key=lambda x: int(x[0].split("@")[1])):
                    file_output.write(f"\t{hitkey:6} = {hitval:.6f}")
                file_output.write("\n")
