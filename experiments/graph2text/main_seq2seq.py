import gc
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data_utils import load_dataset
from src.metrics import build_compute_metrics
from src.utils import can_output_dir_be_used, get_wandb_run
from src.webnlg_preprocessor import WebNLGPreprocessor

np.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)
random.seed(8)


def create_corresponding_docs(ds, cfg):
    assert (
        "document_extractor" in cfg["preprocessor"]
    ), "DocumentExtractor not specified"
    if cfg["preprocessor"]["document_extractor"]["use_extractor_cache"] is False:
        search_engine = instantiate(cfg.preprocessor.search_engine)
        search_engine.indexit(ds["train"])
    else:
        search_engine = None
    document_extractor = instantiate(
        cfg.preprocessor.document_extractor, search_engine=search_engine
    )

    qa_corresponding_prompts = document_extractor.get_documents(
        {key: ds[key] for key in ds if key != "train"},
        batch_size=cfg["preprocessor"]["search_engine"].get("batch_size", 1),
    )
    qa_corresponding_prompts["train"] = [
        None for _ in range(len(ds["train"]))
    ]  # we don't need to search docs for train

    # delete search_engine manually to free gpu memory
    del search_engine
    gc.collect()
    torch.cuda.empty_cache()

    logging.info("Created corresponding_docs")

    return qa_corresponding_prompts


@hydra.main(config_path="configs", config_name="graph2text_seq2seq_mixtral")
def main(cfg: DictConfig):
    print(cfg)
    wandb_run = get_wandb_run(cfg)
    can_output_dir_be_used(cfg)

    ds = load_dataset(cfg["dataset"]["path"])

    config = AutoConfig.from_pretrained(
        cfg["model"]["path"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model"]["path"],
        # device_map="auto",
        config=config,
    )
    # model = torch.compile(model)
    trainer_arguments = Seq2SeqTrainingArguments(**cfg["trainer"])

    sequence_in = cfg["dataset"]["columns"]["sequence_in"]
    sequence_out = cfg["dataset"]["columns"]["sequence_out"]
    max_seq_length = cfg["model"]["tokenizer"]["max_length"]
    max_answer_length = cfg["model"]["trainingArguments"]["max_answer_length"]
    padding = cfg["model"]["trainingArguments"]["padding"]
    preprocessing_num_workers = cfg["model"]["tokenizer"]["num_proc"]
    ignore_pad_token_for_loss = cfg["model"]["trainingArguments"][
        "ignore_pad_token_for_loss"
    ]
    ignore_pad_token = cfg["model"]["trainingArguments"].get("ignore_pad_token", -100)

    delete_empty_lines = cfg["preprocessor"].get("delete_empty_lines", True)
    remove_initial_columns = cfg["preprocessor"].get("remove_initial_columns", True)
    remove_special_chars = cfg["preprocessor"].get("remove_special_chars", True)

    add_prepr_args = {}
    if "search_engine" in cfg["preprocessor"]:
        qa_corresponding_prompts = create_corresponding_docs(ds, cfg)
        add_prepr_args["qa_corresponding_prompts"] = qa_corresponding_prompts

    data_preprocessor = WebNLGPreprocessor(
        dataset=ds,
        tokenizer=tokenizer,
        sequence_in=sequence_in,
        sequence_out=sequence_out,
        max_seq_length=max_seq_length,
        max_answer_length=max_answer_length,
        padding=padding,
        ignore_pad_token_for_loss=ignore_pad_token_for_loss,
        preprocessing_num_workers=preprocessing_num_workers,
        ignore_pad_token=ignore_pad_token,
        start_text_template="convert the [graph] to [text]:",
        **add_prepr_args,
    )

    if trainer_arguments.do_train:
        with trainer_arguments.main_process_first(desc="map pre-processing"):
            train_dataset = data_preprocessor.get_preprocessed_dataset(
                "train",
                delete_empty_lines=delete_empty_lines,
                remove_initial_columns=remove_initial_columns,
                remove_special_chars=remove_special_chars,
            )
            print(train_dataset)
    if trainer_arguments.do_eval:
        with trainer_arguments.main_process_first(desc="map pre-processing"):
            eval_dataset = data_preprocessor.get_preprocessed_dataset(
                "valid",
                delete_empty_lines=delete_empty_lines,
                remove_initial_columns=remove_initial_columns,
                remove_special_chars=remove_special_chars,
            )
    if trainer_arguments.do_predict:
        with trainer_arguments.main_process_first(desc="map pre-processing"):
            predict_dataset = data_preprocessor.get_preprocessed_dataset(
                "test",
                delete_empty_lines=delete_empty_lines,
                remove_initial_columns=remove_initial_columns,
                remove_special_chars=remove_special_chars,
            )

    # Data collator
    label_pad_token_id = (
        ignore_pad_token if ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if trainer_arguments.fp16 else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        train_dataset=train_dataset if trainer_arguments.do_train else None,
        eval_dataset=eval_dataset if trainer_arguments.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, ignore_pad_token),
    )

    if (
        trainer_arguments.save_total_limit
        and cfg["model"]["trainingArguments"]["useEarlyStoppingCallback"]
    ):
        print("add early stopping")
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=trainer_arguments.save_total_limit
        )
        trainer.add_callback(early_stopping)

    # Training
    if trainer_arguments.do_train:
        trainer.train()
        trainer.save_state()

    # Prediction
    if trainer_arguments.do_predict:
        predictions = trainer.predict(predict_dataset, metric_key_prefix="test_eval")

        pred_ids = predictions.predictions
        pred_ids = np.where(
            pred_ids != ignore_pad_token, pred_ids, tokenizer.pad_token_id
        )
        pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        input_ids = np.array(predict_dataset["input_ids"])
        input_ids = np.where(
            input_ids != ignore_pad_token, input_ids, tokenizer.pad_token_id
        )
        sequence_in_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        labels_ids = predictions.label_ids
        labels_ids = np.where(
            labels_ids != ignore_pad_token, labels_ids, tokenizer.pad_token_id
        )
        sequence_out_text = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        results = {
            "data": [
                {
                    sequence_in: sequence_in_text[idx],
                    sequence_out: sequence_out_text[idx],
                    "predicted": predicted,
                }
                for idx, predicted in enumerate(pred_text)
            ],
            "metrics": predictions.metrics,
        }

        if wandb_run is not None:
            wandb_run.log(predictions.metrics)

        print(predictions.metrics)

        with open(Path(cfg["trainer"]["output_dir"]) / "test_results.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(results))

        with open(Path(cfg["trainer"]["output_dir"]) / "run_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    print("YOOOOO")
    main()
