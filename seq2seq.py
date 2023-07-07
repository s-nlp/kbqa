import argparse
import json
import os
import mlflow
import wandb
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from kbqa.config import SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
from kbqa.seq2seq.eval import make_report
from kbqa.seq2seq.train import train as train_seq2seq
from kbqa.seq2seq.utils import (
    dump_eval,
    get_model_logging_dirs,
    load_kbqa_seq2seq_dataset,
    load_mintaka_seq2seq_dataset,
    load_lcquad2_seq2seq_dataset,
    load_model_and_tokenizer_by_name,
)
from kbqa.utils.train_eval import get_best_checkpoint_path
from kbqa.utils import get_default_logger

logger = get_default_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "eval", "train_eval"],
    help="Choose mode for working, train or evaluate/analyze fited model",
)
parser.add_argument(
    "--model_name",
    default="t5-base",
    choices=SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES,
)
parser.add_argument("--dataset_name", default="AmazonScience/mintaka")
parser.add_argument("--dataset_config_name", default="en")
parser.add_argument("--dataset_evaluation_split", default="test")
parser.add_argument("--dataset_cache_dir", default="../datasets/")
parser.add_argument("--save_dir", default="../runs")
parser.add_argument("--run_name", default=None)
parser.add_argument(
    "--lora_on",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="Using LoRA or not (True/False)",
)
parser.add_argument(
    "--lora_r",
    default=64,
    type=int,
    help="LoRA r (int)",
)
parser.add_argument(
    "--lora_alpha",
    default=16,
    type=int,
    help="LoRA Alpha (int)",
)
parser.add_argument(
    "--lora_dropout",
    default=0.05,
    type=float,
    help="LoRA dropout (float)",
)
parser.add_argument(
    "--wandb_on",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="Using WanDB or not (True/False)",
)
parser.add_argument(
    "--mlflow_experiment_name",
    default=None,
    help="Will be used this experiment name if provided",
)
parser.add_argument(
    "--mlflow_run_name", default=None, help="Will be used this run name if provided"
)
parser.add_argument(
    "--mlflow_tracking_uri",
    default="file:///workspace/runs/mlruns",
    help="URI for mlflow tracking",
)
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
parser.add_argument(
    "--num_beams",
    default=200,
    help="Numbers of beams for Beam search (only for eval mode)",
    type=int,
)
parser.add_argument(
    "--num_return_sequences",
    default=200,
    help=(
        "Numbers of return sequencese from Beam search (only for eval mode)."
        " Must be less or equal to num_beams"
    ),
    type=int,
)
parser.add_argument(
    "--num_beam_groups",
    default=20,
    help=(
        "Number of groups to divide num_beams into in order to ensure diversity "
        "among different groups of beams (only for eval mode). "
        "Diverse Beam Search alghoritm "
    ),
    type=int,
)
parser.add_argument(
    "--diversity_penalty",
    default=0.1,
    help=(
        "This value is subtracted from a beam's score if it generates "
        "a token same as any beam from other group at a particular time. "
        "Note that diversity_penalty is only effective if group beam search is enabled."
    ),
    type=float,
)
parser.add_argument(
    "--recall_redirects_on",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="Using WikidataRedirects for calculation recall on evalutaion step, or not.",
)
parser.add_argument(
    "--trainer_mode",
    default="default",
    help=(
        "trainer mode, as a default will used Seq2SeqTrainer, but if provided "
        "Seq2SeqWikidataRedirectsTrainer, that it will used."
    ),
)
parser.add_argument(
    "--apply_redirects_augmentation",
    default=False,
    help="Using Wikidata redirects for augmenting train dataset. Do not use with Seq2SeqWikidataRedirectsTrainer",
    type=lambda x: (str(x).lower() == "true"),
)


def train(args, model_dir, logging_dir):
    model, tokenizer = load_model_and_tokenizer_by_name(args.model_name)

    if args.dataset_name == "AmazonScience/mintaka":
        dataset = load_mintaka_seq2seq_dataset(
            args.dataset_name,
            args.dataset_config_name,
            tokenizer,
        )

    elif args.dataset_name == "s-nlp/lc_quad2":
        dataset = {}
        dataset["train"] = load_lcquad2_seq2seq_dataset(
            args.dataset_name,
            tokenizer,
            args.dataset_cache_dir,
        )

        dataset["validation"] = load_lcquad2_seq2seq_dataset(
            args.dataset_name,
            tokenizer,
            args.dataset_cache_dir,
            split="test",
        )

    else:
        dataset = load_kbqa_seq2seq_dataset(
            args.dataset_name,
            args.dataset_config_name,
            tokenizer,
            args.dataset_cache_dir,
            apply_redirects_augmentation=args.apply_redirects_augmentation,
        )

    if args.lora_on:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)

    if args.wandb_on:
        report_to = "wandb"
    elif args.mlflow_experiment_name is not None:
        report_to = "mlflow"
    else:
        report_to = "tensorboard"

    train_seq2seq(
        run_name=args.run_name,
        report_to=report_to,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        valid_dataset=dataset["validation"],
        output_dir=model_dir,
        logging_dir=logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        trainer_mode=args.trainer_mode,
    )

    if args.lora_on:
        model = model.merge_and_unload()
        model.save_pretrained(Path(model_dir) / "checkpoint-best")

    if args.wandb_on:
        wandb.log(vars(args))


def evaluate(args, model_dir, normolized_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer_by_name(
        args.model_name, get_best_checkpoint_path(model_dir)
    )
    model = model.to(device)

    if args.dataset_name == "AmazonScience/mintaka":
        dataset = load_mintaka_seq2seq_dataset(
            args.dataset_name,
            args.dataset_config_name,
            tokenizer,
            split=args.dataset_evaluation_split,
        )
        label_feature_name = "answerText"
        logger.info(
            f"Eval: MINTAKA Dataset loaded, label_feature_name={label_feature_name}"
        )

    elif args.dataset_name == "s-nlp/lc_quad2":
        dataset = load_lcquad2_seq2seq_dataset(
            args.dataset_name,
            tokenizer,
            args.dataset_cache_dir,
            split="test",
        )
        label_feature_name = "Label"
        logger.info(
            f"Lcquad2.0 Eval: Dataset loaded, label_feature_name={label_feature_name}"
        )

    else:
        dataset = load_kbqa_seq2seq_dataset(
            args.dataset_name,
            args.dataset_config_name,
            tokenizer,
            args.dataset_cache_dir,
            args.dataset_evaluation_split,
            apply_redirects_augmentation=args.apply_redirects_augmentation,
        )
        label_feature_name = "object"
        logger.info(f"Eval: Dataset loaded, label_feature_name={label_feature_name}")

    results_df, report = make_report(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        device=device,
        recall_redirects_on=args.recall_redirects_on,
        label_feature_name=label_feature_name,
    )

    eval_report_dir = dump_eval(results_df, report, args, normolized_model_name)
    if args.mlflow_experiment_name is not None:
        mlflow.log_metrics(report)
        mlflow.log_artifacts(eval_report_dir, "report")
    print(f"Report dumped to {eval_report_dir}")


def validate_args(args):
    if (
        args.apply_redirects_augmentation is True
        and args.trainer_mode == "Seq2SeqWikidataRedirectsTrainer"
    ):
        raise ValueError(
            "Do not use apply_redirects_augmentation with Seq2SeqWikidataRedirectsTrainer - trash data"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    validate_args(args)

    if args.wandb_on:
        os.environ["WANDB_PROJECT"] = "kgqa_seq2seq"

    if args.mlflow_experiment_name is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    model_dir, logging_dir, normolized_model_name = get_model_logging_dirs(
        args.save_dir, args.model_name, args.run_name
    )
    dataset_name = Path(args.dataset_name).name

    if args.mlflow_experiment_name is not None:
        mlflow_experiment_name = args.mlflow_experiment_name
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "True"
        os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.set_experiment_tag("normolized_model_name", normolized_model_name)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params({"args/" + key: value for key, value in vars(args).items()})

    if args.mode == "train":
        if args.mlflow_experiment_name is not None:
            mlflow.set_tag("trained_on", dataset_name)
        train(args, model_dir, logging_dir)

    elif args.mode == "eval":
        if (Path(logging_dir) / "args.json").is_file():
            with open(Path(logging_dir) / "args.json", "r") as file_handler:
                training_args = json.load(file_handler)
            training_dataset_name = Path(training_args["dataset_name"]).name
            if args.mlflow_experiment_name is not None:
                mlflow.set_tag("trained_on", training_dataset_name)

        if args.mlflow_experiment_name is not None:
            mlflow.set_tag("evaluated_on", dataset_name)
        evaluate(args, model_dir, normolized_model_name)

    elif args.mode == "train_eval":
        if args.mlflow_experiment_name is not None:
            mlflow.set_tag("trained_on", dataset_name)
            mlflow.set_tag("evaluated_on", dataset_name)
        train(args, model_dir, logging_dir)
        evaluate(args, model_dir, normolized_model_name)

    else:
        raise ValueError(
            f"Wrong mode argument passed: must be train or eval, passed {args.mode}"
        )

    if args.mlflow_experiment_name is not None:
        mlflow.end_run()

    if args.wandb_on is True:
        wandb.finish()
