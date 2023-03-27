from pathlib import Path
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="t5-large")
parser.add_argument("--model_export_name", default="t5_large")
parser.add_argument(
    "--checkpoint_path",
    default="/mnt/raid/data/kbqa/seq2seq_runs/wdsq_tunned/t5-large/models/checkpoint-7000/",
)
parser.add_argument("--version", default="1.0")

if __name__ == "__main__":
    args = parser.parse_args()

    setup_config = {
        "model_name": args.model_name,
        "mode": "text_generation",
        "do_lower_case": False,
        "num_labels": "0",
        "save_mode": "pretrained",
        "max_length": "64",
        "captum_explanation": False,
        "FasterTransformer": False,
        "embedding_name": args.model_name,
    }
    with open("setup_config.json", "w") as f:
        json.dump(setup_config, f)

    checkpoint_path = Path(args.checkpoint_path)
    SERRIALIZED_FILE = str(checkpoint_path / "pytorch_model.bin")
    EXTRA_FILES = str(checkpoint_path / "config.json") + ",./setup_config.json"

    cmd = (
        "torch-model-archiver "
        f"--model-name {args.model_export_name} "
        f"--version {args.version} "
        f"--serialized-file {SERRIALIZED_FILE} "
        "--handler ./kbqa/seq2seq/transformer_handler_generalized.py "
        f'--extra-files "{EXTRA_FILES}"'
    )

    print(cmd)
    os.system(cmd)
