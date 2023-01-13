import yaml
import pandas as pd
import os
import datasets
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kbqa.utils.train_eval import get_best_checkpoint_path


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

        params["seq2seq"]["dataset"] = os.environ.get(
            "SEQ2SEQ_DATASET", params["seq2seq"]["dataset"]
        )
        params["seq2seq"]["model"]["name"] = os.environ.get(
            "SEQ2SEQ_MODEL_NAME", params["seq2seq"]["model"]["name"]
        )
        params["seq2seq"]["model"]["path"] = os.environ.get(
            "SEQ2SEQ_MODEL_PATH", params["seq2seq"]["model"]["path"]
        )

        params["entity_linking"]["ner"]["path"] = os.environ.get(
            "EL_NER_PATH", params["entity_linking"]["ner"]["path"]
        )

        run_name = os.environ.get("SEQ2SEQ_RUN_NAME", "candidates")

    return params, run_name


def load_datasets(params):
    if params["seq2seq"]["dataset"] == "wdsq":
        test_df = pd.read_csv(
            "./data/wdsq/annotated_wd_data_test_answerable.txt",
            sep="\t",
            names=["S", "P", "O", "Q"],
        )

        train_df = pd.read_csv(
            "./data/wdsq/annotated_wd_data_train_answerable.txt",
            sep="\t",
            names=["S", "P", "O", "Q"],
        )

        valid_df = pd.read_csv(
            "./data/wdsq/annotated_wd_data_valid_answerable.txt",
            sep="\t",
            names=["S", "P", "O", "Q"],
        )

        # train_dataset = datasets.Dataset.from_pandas(train_df)
        train_dataset = None
        valid_dataset = datasets.Dataset.from_pandas(valid_df)
        test_dataset = datasets.Dataset.from_pandas(test_df)
        question_col_name = "Q"
    elif params["seq2seq"]["dataset"] == "rubq":
        dataset = datasets.load_dataset(
            "wikidata_rubq", "multiple_en", cache_dir=".", ignore_verifications=True
        )

        train_dataset = None
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        question_col_name = "question"
    elif params["seq2seq"]["dataset"] == "mintaka":
        with open(
            "/mnt/raid/data/kbqa/datasets/mintaka_one_hop/mintaka_train_simpleq.txt",
            "r",
        ) as f:
            train_ids = [idx.replace("\n", "") for idx in f.readlines()]
        with open(
            "/mnt/raid/data/kbqa/datasets/mintaka_one_hop/mintaka_dev_simpleq.txt", "r"
        ) as f:
            dev_ids = [idx.replace("\n", "") for idx in f.readlines()]
        with open(
            "/mnt/raid/data/kbqa/datasets/mintaka_one_hop/mintaka_test_simpleq.txt", "r"
        ) as f:
            test_ids = [idx.replace("\n", "") for idx in f.readlines()]

        dataset = datasets.load_dataset("AmazonScience/mintaka")
        dataset = dataset.filter(
            lambda r: r["id"] in train_ids + dev_ids + test_ids and r["lang"] == "en"
        )
        # train_dataset = dataset["train"]
        train_dataset = None
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        question_col_name = "question"
    else:
        raise ValueError(
            f"Wrong seq2seq.dataset={params['seq2seq']['dataset']}. Supported only wdsq, rubq, mintaka"
        )

    return train_dataset, valid_dataset, test_dataset, question_col_name


if __name__ == "__main__":
    params, run_name = load_params()
    train_dataset, valid_dataset, test_dataset, question_col_name = load_datasets(
        params
    )

    print(params, run_name)

    tokenizer = AutoTokenizer.from_pretrained(params["seq2seq"]["model"]["name"])
    if params["seq2seq"]["model"]["path"].lower() not in ["none", ""]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            get_best_checkpoint_path(Path(params["seq2seq"]["model"]["path"]))
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            params["seq2seq"]["model"]["name"]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def convert_to_features(example_batch):
        input_encodings = tokenizer(
            example_batch[question_col_name],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        return input_encodings

    def generate_candidate(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(
            dataset, batch_size=params["seq2seq"]["model"]["batch_size"]
        )

        generated_decoded = {
            f"answer_{idx}": []
            for idx in range(params["seq2seq"]["model"]["num_return_sequences"])
        }
        for batch in tqdm(dataloader, desc="evaluate model"):
            generated_ids = model.generate(
                batch["input_ids"].to(device),
                num_beams=params["seq2seq"]["model"]["num_beams"],
                num_return_sequences=params["seq2seq"]["model"]["num_return_sequences"],
                num_beam_groups=params["seq2seq"]["model"]["num_beam_groups"],
                diversity_penalty=params["seq2seq"]["model"]["diversity_penalty"],
            )
            generated_decoded_batch = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            current_batch_size = batch["input_ids"].shape[0]
            for start_batch_idx in range(
                0,
                current_batch_size * params["seq2seq"]["model"]["num_return_sequences"],
                params["seq2seq"]["model"]["num_return_sequences"],
            ):
                for answer_idx, answer in enumerate(
                    generated_decoded_batch[
                        start_batch_idx : start_batch_idx
                        + params["seq2seq"]["model"]["num_return_sequences"]
                    ]
                ):
                    generated_decoded[f"answer_{answer_idx}"].append(answer)

        return generated_decoded

    for split_name, dataset in [
        # ("train", train_dataset),
        # ("valid", valid_dataset),
        ("test", test_dataset),
    ]:
        if dataset is not None:
            Path(
                f"/mnt/raid/data/kbqa/datasets/candidates/{run_name}/{params['seq2seq']['dataset']}/"
            ).mkdir(parents=True, exist_ok=True)
            generated_candidates_df = pd.DataFrame(generate_candidate(dataset))
            generated_candidates_df.to_pickle(
                f"/mnt/raid/data/kbqa/datasets/candidates/{run_name}/{params['seq2seq']['dataset']}/{split_name}.pkl"
            )
