#!/usr/bin/env python3
"""Aggregate evaluation results across multiple runs and calculate mean/std statistics"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import subprocess
import sys

OUTPUTS_DIR = Path(__file__).parent / "outputs"
MINTAKA_EVAL_SCRIPT = Path(__file__).parent.parent.parent.parent / "mintaka_evaluate.py"


def parse_evaluation_file(eval_file: Path) -> dict:
    """Parse evaluation result file and extract metrics"""
    results = {}
    with open(eval_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_category = None
    for line in lines[1:]:  # Skip "Hit scores:" header
        line = line.strip()
        if not line:
            continue
        
        # Extract category name (everything before first tab)
        parts = line.split("\t")
        if not parts:
            continue
        
        category = parts[0].strip()
        metrics = {}
        
        # Extract Hit@N metrics
        for part in parts[1:]:
            match = re.match(r"Hit@(\d+)\s*=\s*([\d.]+)", part)
            if match:
                n = int(match.group(1))
                value = float(match.group(2))
                metrics[f"Hit@{n}"] = value
        
        if metrics:
            results[category] = metrics
    
    return results


def run_evaluation(result_file: Path) -> Path:
    """Run mintaka_evaluate.py on a result file if evaluation doesn't exist"""
    eval_file = OUTPUTS_DIR / f"reranking_result_{result_file.name}.txt"
    
    if not eval_file.exists():
        print(f"Running evaluation for {result_file.name}...")
        cmd = [
            sys.executable,
            str(MINTAKA_EVAL_SCRIPT),
            "--predictions_path",
            str(result_file),
            "--split",
            "test"
        ]
        subprocess.run(cmd, check=True)
    
    return eval_file


def extract_config(result_file: Path) -> tuple:
    """Extract (ds_type, variant) from result filename"""
    name = result_file.name
    # Format: results_r{run}_{ds_type}_rankedby_gptoss20b{_variant}.json
    # ds_type can be: mistral, mixtral, t5largessm, t5xlssm
    match = re.match(r"results_r\d+_([^_]+)_rankedby_gptoss20b(?:_(.+))?\.json", name)
    if match:
        ds_type = match.group(1)
        variant = match.group(2) if match.group(2) else "default"
        return (ds_type, variant)
    return (None, None)


def aggregate_results():
    """Main function to aggregate all evaluation results"""
    result_files = sorted(OUTPUTS_DIR.glob("results_r*_*_rankedby_gptoss20b*.json"))
    
    # Group results by configuration
    config_results = defaultdict(lambda: defaultdict(list))
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        ds_type, variant = extract_config(result_file)
        if ds_type is None:
            print(f"Warning: Could not parse config from {result_file.name}")
            continue
        
        # Run evaluation if needed
        eval_file = run_evaluation(result_file)
        
        # Parse evaluation results
        try:
            results = parse_evaluation_file(eval_file)
            config_results[(ds_type, variant)]["runs"].append(results)
        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
            continue
    
    # Calculate statistics
    aggregated = {}
    for (ds_type, variant), data in config_results.items():
        runs = data["runs"]
        num_runs = len(runs)
        if num_runs != 3:
            print(f"Warning: Expected 3 runs for {ds_type}/{variant}, found {num_runs}")
        
        # Aggregate metrics across runs
        config_key = f"{ds_type}_{variant}"
        aggregated[config_key] = {
            "metadata": {
                "ds_type": ds_type,
                "variant": variant,
                "num_runs": num_runs
            },
            "results": {}
        }
        
        # Get all categories from first run
        categories = set()
        for run in runs:
            categories.update(run.keys())
        
        for category in sorted(categories):
            # Collect values for each Hit@N across runs
            hit_metrics = defaultdict(list)
            for run in runs:
                if category in run:
                    for hit_key, value in run[category].items():
                        hit_metrics[hit_key].append(value)
            
            # Calculate mean and std
            category_stats = {}
            for hit_key in sorted(hit_metrics.keys(), key=lambda x: int(x.split("@")[1])):
                values = hit_metrics[hit_key]
                if values:
                    category_stats[f"{hit_key}_mean"] = np.mean(values)
                    category_stats[f"{hit_key}_std"] = np.std(values)
            
            if category_stats:
                aggregated[config_key]["results"][category] = category_stats
    
    return aggregated


def print_results(aggregated: dict):
    """Print aggregated results in a readable format"""
    output_file = OUTPUTS_DIR / "aggregated_results_rankedby_gptoss20b_mean_std.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Aggregated Results (Mean ± Std across runs)\n")
        f.write("=" * 80 + "\n\n")
        
        for config_key in sorted(aggregated.keys()):
            config_data = aggregated[config_key]
            metadata = config_data["metadata"]
            results = config_data["results"]
            
            ds_type = metadata["ds_type"]
            variant = metadata["variant"]
            num_runs = metadata["num_runs"]
            
            f.write(f"\nConfiguration: ds_type={ds_type}, type={variant}, runs={num_runs}\n")
            f.write("-" * 80 + "\n")
            
            for category in sorted(results.keys()):
                f.write(f"\n{category}:\n")
                metrics = results[category]
                
                # Group by Hit@N
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
    
    print(f"\nResults saved to {output_file}")
    
    # Also print to console
    with open(output_file, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    aggregated = aggregate_results()
    print_results(aggregated)

