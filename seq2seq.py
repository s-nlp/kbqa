import argparse
import json
import torch
from pathlib import Path
from config import SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
from utils import get_best_checkpoint_path
from seq2seq.train import train as train_seq2seq
from seq2seq.eval import make_report
from seq2seq.utils import (
    hf_model_name_mormolize,
    load_kbqa_seq2seq_dataset,
    load_model_and_tokenizer_by_name,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=["train", "eval"],
    help="Choose mode for working, train or evaluate/analyze fited model",
)
parser.add_argument(
    "model_name",
    choices=SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES,
)
parser.add_argument("--dataset_name", default="../wikidata_simplequestions/")
parser.add_argument("--dataset_config_name", default="answerable_en")
parser.add_argument("--dataset_cache_dir", default="../datasets/")
parser.add_argument("--save_dir", default="../runs")
parser.add_argument(
    "--num_train_epochs",
    default=8,
    type=int,
)
parser.add_argument(
    "--per_device_train_batch_size",
    default=1,
    type=int,
)
parser.add_argument(
    "--logging_steps",
    default=500,
    type=int,
)
parser.add_argument(
    "--eval_steps",
    default=500,
    type=int,
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=8,
    type=int,
)


def get_model_logging_dirs(save_dir, model_name):
    normolized_model_name = hf_model_name_mormolize(model_name)
    model_dir = Path(save_dir) / "models" / normolized_model_name
    logging_dir = Path(save_dir) / "logs" / normolized_model_name

    return model_dir, logging_dir, normolized_model_name


def train(args):
    output_dir, logging_dir, _ = get_model_logging_dirs(args.save_dir, args.model_name)

    train_seq2seq(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_cache_dir=args.dataset_cache_dir,
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )


def evaluate(args):
    output_dir, normolized_model_name, _ = get_model_logging_dirs(
        args.save_dir, args.model_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer_by_name(
        args.model_name, get_best_checkpoint_path(output_dir)
    )
    model = model.to(device)

    dataset = load_kbqa_seq2seq_dataset(
        args.dataset_name,
        args.dataset_config_name,
        tokenizer,
        args.dataset_cache_dir,
        "test",
    )

    results_df, report = make_report(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        device=device,
    )

    eval_report_dir = Path(args.save_dir) / "evaluation" / normolized_model_name.name
    eval_report_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(eval_report_dir / "results.csv", index=False)
    with open(eval_report_dir / "report.json", "w", encoding=None) as file_handler:
        json.dump(report, file_handler)

    print(report)
    print(f"Report dumped to {eval_report_dir}")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError(
            f"Wrong mode argument passed: must be train or eval, passed {args.mode}"
        )
