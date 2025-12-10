"""Upload CSV results to HuggingFace as a subset"""
import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict

parse = argparse.ArgumentParser()
parse.add_argument(
    "--outputs_train_path",
    type=str,
    default=None,
    help="Path to train CSV file",
)

parse.add_argument(
    "--outputs_val_path",
    type=str,
    default=None,
    help="Path to validation CSV file",
)

parse.add_argument(
    "--outputs_test_path",
    type=str,
    default=None,
    help="Path to test CSV file",
)

parse.add_argument(
    "--hf_path",
    type=str,
    default="s-nlp/MKQASubgraphsRanking",
    help="Path to upload to HuggingFace",
)

parse.add_argument(
    "--subset_name",
    type=str,
    default="mkqa_t5largessm",
    help="Name for the subset when pushing to HuggingFace. Subset will be named 'name_outputs'.",
)


if __name__ == "__main__":
    args = parse.parse_args()

    # Load CSV files
    train_df = None
    val_df = None
    test_df = None

    if args.outputs_train_path:
        if not os.path.exists(args.outputs_train_path):
            raise ValueError(f"Train CSV file not found: {args.outputs_train_path}")
        train_df = pd.read_csv(args.outputs_train_path)

    if args.outputs_val_path:
        if not os.path.exists(args.outputs_val_path):
            raise ValueError(f"Validation CSV file not found: {args.outputs_val_path}")
        val_df = pd.read_csv(args.outputs_val_path)

    if args.outputs_test_path:
        if not os.path.exists(args.outputs_test_path):
            raise ValueError(f"Test CSV file not found: {args.outputs_test_path}")
        test_df = pd.read_csv(args.outputs_test_path)

    if train_df is None and val_df is None and test_df is None:
        raise ValueError(
            "At least one of --outputs_train_path/--outputs_val_path/--outputs_test_path must be provided"
        )

    # Upload to HF
    ds = DatasetDict()
    if train_df is not None:
        ds["train"] = Dataset.from_pandas(train_df)
    if val_df is not None:
        ds["validation"] = Dataset.from_pandas(val_df)
    if test_df is not None:
        ds["test"] = Dataset.from_pandas(test_df)

    if len(ds) > 0:
        subset_name = f"{args.subset_name}_outputs"
        try:
            ds.push_to_hub(args.hf_path, config_name=subset_name)
        except (TypeError, ValueError):
            ds.push_to_hub(args.hf_path)
            print(f"Note: Pushed dataset without config_name. Subset name '{subset_name}' is for reference only.")

