""" late fusion script"""
from argparse import ArgumentParser, RawTextHelpFormatter
import json
from tqdm import tqdm
from datasets import load_dataset
from kbqa.wikidata import query_wikidata_label_to_entity as lab2ent


# pylint: disable=trailing-whitespace
DESCRIPTION = """Script to apply late fusion to the reranked answer candidates.

New fusion ranks will be an average of the original ranking (from vanilla language model)
and the new rankings (from different reranking methods ie. Linear Regression, Logistic Regression etc.)
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
    help="Path to JSONL file with reranking predictions" + EXAMPLE_OF_DATA_FORMAT,
    default="/workspace/storage/misc/subgraphs_reranking_runs/reranking_model_results/mistral/mpnet_nohl_gap_reranking_results.jsonl",
)

parser.add_argument(
    "--kgqa_ds_path",
    type=str,
    default="s-nlp/KGQA_Subgraphs_Ranking",
    help="Path to train sequence data file (HF)",
)

parser.add_argument(
    "--ds_type",
    type=str,
    default="mistral",
    choices=["t5largessm", "t5xlssm", "mistral", "mixtral"],
    help="dataset types, can be t5large, t5xl, mistral or mixtral",
)

parser.add_argument(
    "--hf_cache_dir",
    type=str,
    default="/workspace/storage/misc/huggingface",
    help="huggingface cache directory",
)

parser.add_argument(
    "--output_path",
    type=str,
    default="/workspace/storage/misc/huggingface",
    help="path for the results folder ",
)


def find_ranking_position(ent_id, original_ranks):
    """given the list of original ranks, find the rank position of ent id"""
    for rank_pos, og_ent_id in enumerate(original_ranks):
        for og_ent_id_alias in og_ent_id:  # could be more than 1 alias
            if ent_id == og_ent_id_alias:
                return rank_pos

    return None


def apply_fusion_ranking(ranked_predictions, base_model_preditions):
    """baseline late fusion ranking, avg or original and ranked ranking
    ranked_predictions is the new ranking from one of the approach,
    base model prediction is the ranking from the vanilla base model"""
    final_fusion = []  # iterating throught the new ranked prediction
    for ranked_prediction in tqdm(ranked_predictions, desc="processing fusion...."):
        # get the orinal rank from base model
        curr_base_model_pred = base_model_preditions[
            base_model_preditions["id"] == ranked_prediction["QuestionID"]
        ]
        original_ranks = curr_base_model_pred.drop(["id", "question", "target"], axis=1)
        original_ranks = list(
            original_ranks.values[0]
        )  # original base model rank as list
        original_ranks = [lab2ent.label_to_entity(label) for label in original_ranks]
        candidate_fusion_dict = {}  # building the new ranking

        # ranked ans is the same as base model (yesno, count or questions w/ no subgraphs)
        if ranked_prediction["RankedAnswers"][0]["AnswerEntityID"] is None:
            candidate_fusion_dict = ranked_prediction  # no changes to the ranking
        else:  # ranked predictions is different than base model prediction -> apply fusion
            candidate_fusion_dict = {}  # building the new ranking
            candidate_fusion_dict["QuestionID"] = ranked_prediction["QuestionID"]

            # iterate through the current ranked candidate, find fusion ranking for each candidate
            fusion_rank_dict = {}
            for ranked_idx, ranked_ans in enumerate(ranked_prediction["RankedAnswers"]):
                original_rank_idx = find_ranking_position(
                    ranked_ans["AnswerEntityID"], original_ranks
                )
                if original_rank_idx:
                    fusion_rank = (ranked_idx + original_rank_idx) / 2
                else:  # if ranking doesn't exist in seq2seq outputs, no fusion
                    fusion_rank = ranked_idx
                fusion_rank_dict[ranked_ans["AnswerEntityID"]] = fusion_rank

            # sort the fusion rank dict from lowest rank (best) to highest rank (worst)
            fusion_rank_dict = dict(
                sorted(fusion_rank_dict.items(), key=lambda item: item[1])
            )
            # building the new reshuffuled answer candidates with fusion ranking
            candidate_fusion_ranks = []
            for fusion_cand in list(fusion_rank_dict.keys()):
                candidate_fusion_ranks.append(
                    {
                        "AnswerEntityID": fusion_cand,
                        "AnswerString": None,
                        "Score": fusion_rank_dict[fusion_cand],
                    }
                )
            candidate_fusion_dict["RankedAnswers"] = candidate_fusion_ranks

        final_fusion.append(candidate_fusion_dict)

    return final_fusion


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.predictions_path, "r", encoding="utf-8") as f:
        reranking_predictions = [json.loads(line) for line in f.readlines()]

    llm_outputs = load_dataset(
        args.kgqa_ds_path,
        data_dir=f"{args.ds_type}_outputs",
        cache_dir=args.hf_cache_dir,
    )["test"].to_pandas()

    fusion_reranking_predictions = apply_fusion_ranking(
        reranking_predictions, llm_outputs
    )
    pred_path_split = args.predictions_path.split(".")
    output_path = f"{pred_path_split[0]}_fusion.{pred_path_split[1]}"

    with open(output_path, "w+", encoding="utf-8") as file:
        for result in fusion_reranking_predictions:
            file.write(json.dumps(result) + "\n")
