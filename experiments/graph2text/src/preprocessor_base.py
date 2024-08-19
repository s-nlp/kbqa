from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class DataPreprocessorBase:
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        sequence_in: str,
        sequence_out: str,
        preprocessing_num_workers: int = 32,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_in = sequence_in
        self.sequence_out = sequence_out
        self.preprocessing_num_workers = preprocessing_num_workers

    def get_preprocessed_dataset(
        self,
        subset_name="train",
        delete_empty_lines=True,
        remove_initial_columns=True,
        remove_special_chars=True,
    ) -> Dataset | DatasetDict:
        dataset = self.dataset[subset_name]

        if delete_empty_lines:
            dataset = dataset.filter(
                lambda example: example[self.sequence_out] is not None
            )
            dataset = dataset.filter(
                lambda example: len(example[self.sequence_out].strip()) != 0
            )

        if remove_special_chars:
            dataset = dataset.map(
                self._remove_special_chars_map_fn,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                desc=f"Removing special characters in {subset_name} dataset",
            )

        if remove_initial_columns is True:
            column_names = dataset.column_names
        else:
            column_names = []

        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            desc=f"Running tokenizer on {subset_name} dataset",
        )

        return dataset

    def _preprocess_function(self, *args, **kwargs):
        raise NotImplementedError(
            "Define _preprocess_function in %s." % (self.__class__.__name__)
        )

    def _remove_special_chars(self, string):
        string = string.replace("\r", "")
        return " ".join([item for item in string.split(" ") if item != ""])

    def _remove_special_chars_map_fn(self, examples):
        for column_name in [self.sequence_in, self.sequence_out]:
            examples[column_name] = [
                self._remove_special_chars(string) for string in examples[column_name]
            ]
        return examples
