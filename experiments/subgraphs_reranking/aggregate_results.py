#!/usr/bin/env python3
"""Aggregate evaluation results across multiple runs and calculate mean/std statistics"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import subprocess
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description="Aggregate evaluation results across multiple runs")
parser.add_argument(
    "--ds_type",
    type=str,
    default=None,
    help="Filter results by dataset type (e.g., t5largessm, t5xlssm, mistral, mixtral). If not specified, processes all dataset types."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mintaka",
    choices=["mintaka", "mkqa-hf"],
    help="Dataset to use: mintaka or mkqa-hf."
)

MINTAKA_EVAL_SCRIPT = Path(__file__).parent.parent.parent / "mintaka_evaluate.py"
MKQA_EVAL_SCRIPT = Path(__file__).parent.parent.parent / "mkqa_evaluate.py"


def get_results_base_dir(dataset: str) -> Path:
    """Get the results base directory based on dataset type"""
    if dataset == "mkqa-hf":
        return Path("./reranking_model_results/mkqa-hf")
    return Path("./reranking_model_results/mintaka")


def parse_evaluation_file(eval_file: Path) -> dict:
    """Parse evaluation result file and extract metrics"""
    results = {}
    with open(eval_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip "Hit scores:" header
        line = line.strip()
        if not line:
            continue
        
        parts = line.split("\t")
        if not parts:
            continue
        
        category = parts[0].strip()
        metrics = {}
        
        for part in parts[1:]:
            match = re.match(r"Hit@(\d+)\s*=\s*([\d.]+)", part)
            if match:
                n = int(match.group(1))
                value = float(match.group(2))
                metrics[f"Hit@{n}"] = value
        
        if metrics:
            results[category] = metrics
    
    return results


def run_evaluation(result_file: Path, dataset: str) -> Path:
    """Run evaluation script on a result file if evaluation doesn't exist"""
    eval_file = result_file.parent / f"reranking_result_{result_file.name}.txt"
    
    if not eval_file.exists():
        print(f"Running evaluation for {result_file.name}...")
        eval_script = MKQA_EVAL_SCRIPT if dataset == "mkqa-hf" else MINTAKA_EVAL_SCRIPT
        cmd = [
            sys.executable,
            str(eval_script),
            "--predictions_path",
            str(result_file),
            "--split",
            "test"
        ]
        subprocess.run(cmd, check=True)
    
    return eval_file


def extract_config(result_file: Path) -> tuple:
    """Extract (model_type, feature_combo, ds_type, run) from result filename
    
    Handles two patterns:
    1. With run: {model_type}_{feature_combo}_reranking_seq2seq_run_{run}_{ds_type}_results.jsonl
    2. Without run (baselines): {model_type}_{feature_combo}_reranking_seq2seq_{ds_type}_results.jsonl
    """
    name = result_file.name
    # Pattern 1: With run number (multi-run experiments)
    match = re.match(
        r"([^_]+)_(.+)_reranking_seq2seq_run_(\d+)_(.+?)_results\.jsonl",
        name
    )
    if match:
        model_type = match.group(1)
        feature_combo = match.group(2)
        run = int(match.group(3))
        ds_type = match.group(4)
        return (model_type, feature_combo, ds_type, run)
    
    # Pattern 2: Without run number (baseline methods, single run)
    # Handle cases like: NO_reranking_seq2seq_{ds_type}_results.jsonl
    # or: full_random_reranking_seq2seq_{ds_type}_results.jsonl
    # or: logreg_text_reranking_seq2seq_{ds_type}_results.jsonl
    match = re.match(
        r"(.+?)_reranking_seq2seq_(.+?)_results\.jsonl",
        name
    )
    if match:
        prefix = match.group(1)  # Everything before "_reranking_seq2seq_"
        ds_type = match.group(2)
        
        # Known model types that have feature combos (split on first underscore)
        models_with_features = {"logreg", "linreg", "mpnet", "catboost"}
        
        # Try to split into model_type and feature_combo
        parts = prefix.split("_", 1)
        if len(parts) == 2 and parts[0] in models_with_features:
            # Has feature combo: e.g., "logreg_text" -> model_type="logreg", feature_combo="text"
            model_type = parts[0]
            feature_combo = parts[1]
        else:
            # No feature combo or unknown model: e.g., "NO", "full_random", "semantic_mpnet"
            # The whole prefix is the model_type
            model_type = prefix
            feature_combo = ""
        
        # Use run=0 for baselines to distinguish from multi-run experiments
        return (model_type, feature_combo, ds_type, 0)
    
    return (None, None, None, None)


def aggregate_results(ds_type_filter: str = None, dataset: str = "mintaka"):
    """Main function to aggregate all evaluation results
    
    Parameters
    ----------
    ds_type_filter : str, optional
        If provided, only process results for this dataset type
    dataset : str
        Dataset type: "mintaka" or "mkqa-hf"
    """
    results_base_dir = get_results_base_dir(dataset)
    
    if not results_base_dir.exists():
        print(f"Warning: Results directory {results_base_dir} does not exist")
        return {}
    
    result_files = []
    for ds_dir in results_base_dir.iterdir():
        if ds_dir.is_dir():
            if ds_type_filter is not None and ds_dir.name != ds_type_filter:
                continue
            # Find files with run numbers (multi-run experiments)
            result_files.extend(
                ds_dir.glob("*_reranking_seq2seq_run_*_results.jsonl")
            )
            # Find files without run numbers (baseline methods)
            # Exclude files that already matched the pattern above
            baseline_files = [
                f for f in ds_dir.glob("*_reranking_seq2seq_*_results.jsonl")
                if "_run_" not in f.name
            ]
            result_files.extend(baseline_files)
    
    result_files = sorted(result_files)
    
    config_results = defaultdict(lambda: defaultdict(list))
    
    print(f"Found {len(result_files)} result files" + (f" for ds_type={ds_type_filter}" if ds_type_filter else ""))
    
    for result_file in result_files:
        model_type, feature_combo, ds_type, run = extract_config(result_file)
        if model_type is None:
            print(f"Warning: Could not parse config from {result_file.name}")
            continue
        
        if ds_type_filter is not None and ds_type != ds_type_filter:
            continue
        
        eval_file = run_evaluation(result_file, dataset)
        
        try:
            results = parse_evaluation_file(eval_file)
            config_key = (model_type, feature_combo, ds_type)
            config_results[config_key]["runs"].append((run, results))
        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
            continue
    
    aggregated = {}
    for (model_type, feature_combo, ds_type), data in config_results.items():
        runs_data = sorted(data["runs"], key=lambda x: x[0])
        runs = [r[1] for r in runs_data]
        num_runs = len(runs)
        
        # Check if this is a baseline (run=0) or multi-run experiment
        is_baseline = any(r[0] == 0 for r in runs_data)
        
        if not is_baseline and num_runs != 3:
            print(f"Warning: Expected 3 runs for {model_type}/{feature_combo}/{ds_type}, found {num_runs}")
        
        config_key = f"{model_type}_{feature_combo}_{ds_type}"
        aggregated[config_key] = {
            "metadata": {
                "model_type": model_type,
                "feature_combo": feature_combo,
                "ds_type": ds_type,
                "num_runs": num_runs,
                "is_baseline": is_baseline
            },
            "results": {}
        }
        
        categories = set()
        for run in runs:
            categories.update(run.keys())
        
        for category in sorted(categories):
            hit_metrics = defaultdict(list)
            for run in runs:
                if category in run:
                    for hit_key, value in run[category].items():
                        hit_metrics[hit_key].append(value)
            
            category_stats = {}
            for hit_key in sorted(hit_metrics.keys(), key=lambda x: int(x.split("@")[1])):
                values = hit_metrics[hit_key]
                if values:
                    if is_baseline and num_runs == 1:
                        # For baselines (single run), mean = value, std = 0.0
                        category_stats[f"{hit_key}_mean"] = values[0]
                        category_stats[f"{hit_key}_std"] = 0.0
                    else:
                        # For multi-run experiments, calculate mean and std
                        category_stats[f"{hit_key}_mean"] = np.mean(values)
                        category_stats[f"{hit_key}_std"] = np.std(values)
            
            if category_stats:
                aggregated[config_key]["results"][category] = category_stats
    
    return aggregated


def print_results(aggregated: dict, ds_type_filter: str = None, dataset: str = "mintaka"):
    """Print aggregated results in a readable format
    
    Parameters
    ----------
    aggregated : dict
        Aggregated results dictionary
    ds_type_filter : str, optional
        Dataset type filter used, for output filename
    dataset : str
        Dataset type: "mintaka" or "mkqa-hf"
    """
    results_base_dir = get_results_base_dir(dataset)
    
    if ds_type_filter:
        output_file = results_base_dir / f"aggregated_results_mean_std_{ds_type_filter}.txt"
    else:
        output_file = results_base_dir / "aggregated_results_mean_std.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Aggregated Results (Mean ± Std across runs)\n")
        f.write("=" * 80 + "\n\n")
        
        for config_key in sorted(aggregated.keys()):
            config_data = aggregated[config_key]
            metadata = config_data["metadata"]
            results = config_data["results"]
            
            model_type = metadata["model_type"]
            feature_combo = metadata["feature_combo"]
            ds_type = metadata["ds_type"]
            num_runs = metadata["num_runs"]
            is_baseline = metadata.get("is_baseline", False)
            
            run_info = f"runs={num_runs}" if not is_baseline else "baseline (single run)"
            f.write(f"\nConfiguration: model={model_type}, features={feature_combo}, ds_type={ds_type}, {run_info}\n")
            f.write("-" * 80 + "\n")
            
            for category in sorted(results.keys()):
                f.write(f"\n{category}:\n")
                metrics = results[category]
                
                hit_n_values = defaultdict(dict)
                for key, value in metrics.items():
                    if "_mean" in key:
                        hit_n = key.replace("_mean", "")
                        hit_n_values[hit_n]["mean"] = value
                    elif "_std" in key:
                        hit_n = key.replace("_std", "")
                        hit_n_values[hit_n]["std"] = value
                
                for hit_n in sorted(hit_n_values.keys(), key=lambda x: int(x.split("@")[1])):
                    mean = hit_n_values[hit_n]["mean"]
                    std = hit_n_values[hit_n]["std"]
                    f.write(f"  {hit_n:8} = {mean:.4f} ± {std:.4f}\n")
            
            f.write("\n")
        
        # Calculate and print upper bound (maximum Hit@K values across all configurations)
        f.write("\n" + "=" * 80 + "\n")
        f.write("Upper Bound (Maximum Hit@K across all configurations)\n")
        f.write("=" * 80 + "\n\n")
        
        # Collect all categories and Hit@K metrics (Hit@1 to Hit@30)
        all_categories = set()
        all_hit_metrics = defaultdict(lambda: defaultdict(list))
        
        for config_key, config_data in aggregated.items():
            results = config_data["results"]
            for category, metrics in results.items():
                all_categories.add(category)
                for key, value in metrics.items():
                    if "_mean" in key:
                        hit_n = key.replace("_mean", "")
                        # Only include Hit@1 to Hit@30
                        hit_k = int(hit_n.split("@")[1])
                        if 1 <= hit_k <= 30:
                            all_hit_metrics[category][hit_n].append(value)
        
        # Print maximum values for each category and Hit@K
        for category in sorted(all_categories):
            f.write(f"{category}:\n")
            hit_metrics = all_hit_metrics[category]
            
            for hit_n in sorted(hit_metrics.keys(), key=lambda x: int(x.split("@")[1])):
                max_value = max(hit_metrics[hit_n])
                f.write(f"  {hit_n:8} = {max_value:.4f}\n")
            
            f.write("\n")
    
    print(f"\nResults saved to {output_file}")
    
    with open(output_file, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    args = parser.parse_args()
    aggregated = aggregate_results(args.ds_type, args.dataset)
    print(f"Aggregated results: {aggregated}")
    print_results(aggregated, args.ds_type, args.dataset)

