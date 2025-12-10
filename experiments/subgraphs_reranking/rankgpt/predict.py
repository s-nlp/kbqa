#!/usr/bin/env python3
"""CLI script for running RankGPT predictions on KGQA data"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import ranking_data_utils
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from ranking_data_utils import prepare_data
from rankgpt_ranker import RankGPTRanker


def main():
    parser = argparse.ArgumentParser(description="Run RankGPT ranking on KGQA data")
    
    # Required arguments
    parser.add_argument("--model_name", required=True, help="Model name for ranking (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--ds_type", required=True, choices=["t5largessm", "t5xlssm", "mistral", "mixtral"], 
                       help="Dataset type to use")
    parser.add_argument("--dataset", type=str, default="mintaka", choices=["mintaka", "mkqa-hf"],
                       help="Dataset to use: mintaka or mkqa-hf.")
    parser.add_argument("--output_path", required=True, help="Path to save results JSONL file")
    
    # Optional arguments
    parser.add_argument("--window_size", type=int, default=20, help="Window size for sliding window (default: 20)")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for sliding window (default: 10)")
    parser.add_argument("--api_base", help="API base URL (overrides OPENAI_BASE_URL env var)")
    parser.add_argument("--api_key", help="API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for API calls (default: 3)")
    parser.add_argument("--retry_delay", type=float, default=1.0, help="Delay between retries in seconds (default: 1.0)")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], 
                       help="Dataset split to use (default: test)")
    parser.add_argument("--graph_sequence_feature", choices=["highlighted_determ_sequence", "no_highlighted_determ_sequence"], 
                       help="Optional graph sequence feature to include in prompts (default: None)")
    
    args = parser.parse_args()
    
    # Validate API configuration
    api_base = args.api_base or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_base:
        print("Error: API base URL must be provided via --api_base or OPENAI_BASE_URL environment variable")
        sys.exit(1)
    if not api_key:
        print("Error: API key must be provided via --api_key or OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print(f"Using API base: {api_base}")
    print(f"Using model: {args.model_name}")
    print(f"Using dataset: {args.dataset}")
    print(f"Using dataset type: {args.ds_type}")
    print(f"Using split: {args.split}")
    print(f"Window size: {args.window_size}")
    print(f"Step size: {args.step_size}")
    print(f"Graph sequence feature: {args.graph_sequence_feature or 'None'}")
    
    # Load datasets
    print("Loading datasets...")
    try:
        if args.dataset == "mintaka":
            base_dataset_path = "AmazonScience/mintaka"
            kgqa_ds_path = "s-nlp/KGQASubgraphsRanking"
            features_data_dir = f"{args.ds_type}_subgraphs"
            outputs_data_dir = f"{args.ds_type}_outputs"
        elif args.dataset == "mkqa-hf":
            base_dataset_path = "Dms12/mkqa_mintaka_format_with_question_entities"
            kgqa_ds_path = "s-nlp/MKQASubgraphsRanking"
            features_data_dir = f"mkqa_{args.ds_type}_subgraphs"
            outputs_data_dir = f"mkqa_{args.ds_type}_outputs"
        
        # Load subgraph features
        features_ds = load_dataset(
            kgqa_ds_path,
            data_dir=features_data_dir,
        )
        
        # Load model outputs
        outputs_ds = load_dataset(
            kgqa_ds_path,
            data_dir=outputs_data_dir,
        )
        
        # Load base dataset
        if args.dataset == "mintaka":
            base_ds = load_dataset(base_dataset_path, revision="refs/convert/parquet", data_dir=f"en")
        else:
            base_ds = load_dataset(base_dataset_path)
        print("Datasets loaded successfully")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)
    
    # Prepare data
    print("Preparing data...")
    test_df = prepare_data(
        base_ds[args.split], 
        outputs_ds[args.split], 
        features_ds[args.split]
    )
    print(f"Prepared {len(test_df)} rows of data")
    print(test_df.head())
        
    # Initialize RankGPT ranker
    print("Initializing RankGPT ranker...")
    try:
        ranker = RankGPTRanker(
            api_base=api_base,
            api_key=api_key,
            model_name=args.model_name,
            window_size=args.window_size,
            step_size=args.step_size,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            graph_sequence_feature=args.graph_sequence_feature
        )
        print("RankGPT ranker initialized successfully")
        
    except Exception as e:
        print(f"Error initializing ranker: {e}")
        sys.exit(1)
    
    # Run ranking
    print("Running ranking...")
    results = ranker.rerank(test_df)
    print(f"Ranking completed for {len(results)} questions")
        
    
    # Save results
    print(f"Saving results to {args.output_path}...")
    try:
        # Create output directory if it doesn't exist
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(f"Results saved successfully to {args.output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    print("RankGPT ranking completed successfully!")


if __name__ == "__main__":
    main()
