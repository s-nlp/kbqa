from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class DataPreprocessorBase:
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        question_column: str,
        answer_column: str,
        preprocessing_num_workers: int = 32,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.question_column = question_column
        self.answer_column = answer_column
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
                lambda example: example[self.answer_column] is not None
            )
            dataset = dataset.filter(
                lambda example: len(example[self.answer_column].strip()) != 0
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
        for column_name in [self.question_column, self.answer_column]:
            examples[column_name] = [
                self._remove_special_chars(string) for string in examples[column_name]
            ]
        return examples


class QuestionAnswerPreprocessor(DataPreprocessorBase):
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        question_column: str,
        answer_column: str,
        max_seq_length: int,
        max_answer_length: int,
        padding: str,
        ignore_pad_token_for_loss: bool = True,
        ignore_pad_token=-100,
        preprocessing_num_workers: int = 32,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            question_column=question_column,
            answer_column=answer_column,
            preprocessing_num_workers=preprocessing_num_workers,
        )

        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.ignore_pad_token = ignore_pad_token

    def _preprocess_function(
        self,
        examples,
    ):
        inputs, targets = examples[self.question_column], examples[self.answer_column]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )
        # Tokenize targets with text_target=...
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_answer_length,
            padding=self.padding,
            truncation=True,
        )
        # If we are padding here, replace all tokenizer.pad_token_id
        # in the labels by ignore_pad_token when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (
                        lbl
                        if lbl != self.tokenizer.pad_token_id
                        else self.ignore_pad_token
                    )
                    for lbl in label
                ]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class PromptQustAnswPreprocessor(QuestionAnswerPreprocessor):
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        question_column: str,
        answer_column: str,
        prompt_string: str,
        prompt_start_string: str,
        max_seq_length: int,
        max_answer_length: int,
        padding: str,
        ignore_pad_token_for_loss: bool = True,
        preprocessing_num_workers: int = 32,
        ignore_pad_token=-100,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            question_column=question_column,
            answer_column=answer_column,
            max_seq_length=max_seq_length,
            max_answer_length=max_answer_length,
            padding=padding,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            preprocessing_num_workers=preprocessing_num_workers,
            ignore_pad_token=ignore_pad_token,
        )

        self.prompt_string = prompt_string
        self.prompt_start_string = prompt_start_string

    def _preprocess_function(self, examples):
        inputs, targets = examples[self.question_column], examples[self.answer_column]
        inputs = list(
            map(
                lambda x: self.prompt_string + "\n" + self.prompt_start_string + x,
                inputs,
            )
        )

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )
        # Tokenize targets with text_target=...
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_answer_length,
            padding=self.padding,
            truncation=True,
        )
        # If we are padding here, replace all tokenizer.pad_token_id
        # in the labels by ignore_pad_token when we want to ignore
        # padding in the loss.

        self._pad_input_ids(labels)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _pad_input_ids(self, labels):
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (
                        lbl
                        if lbl != self.tokenizer.pad_token_id
                        else self.ignore_pad_token
                    )
                    for lbl in label
                ]
                for label in labels["input_ids"]
            ]
        return labels


class SearchPromptQustAnswPreprocessor(PromptQustAnswPreprocessor):
    """
    prompt_template format:
    '<START_PROMPT>{CORRESPONDING_DOC}<END_PROMPT>{QUESTION}\n<ANSWER_START_STRING>'
    For examle, 'Примеры пар: {} Вопрос: {}\nОтвет: '
    """

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        question_column: str,
        answer_column: str,
        prompt_template: str,
        qa_corresponding_prompts: dict[str, list[str]],
        max_seq_length: int,
        max_answer_length: int,
        padding: str,
        ignore_pad_token_for_loss: bool = True,
        preprocessing_num_workers: int = 32,
        ignore_pad_token=-100,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            question_column=question_column,
            answer_column=answer_column,
            max_seq_length=max_seq_length,
            max_answer_length=max_answer_length,
            padding=padding,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            preprocessing_num_workers=preprocessing_num_workers,
            ignore_pad_token=ignore_pad_token,
            prompt_string=None,
            prompt_start_string=None,
        )

        for key in dataset.keys():
            assert (
                key in qa_corresponding_prompts
            ), f"Key {key} not in keys from qa_corresponding_prompts"
            assert len(dataset[key]) == len(
                qa_corresponding_prompts[key]
            ), "Len of dataset[{}] is {} != Len of corresponding_docs[{}] ({})".format(
                key, len(dataset[key]), key, len(qa_corresponding_prompts[key])
            )

        self.prompt_template = prompt_template
        self.qa_corresponding_prompts = qa_corresponding_prompts

    def get_preprocessed_dataset(
        self,
        subset_name="train",
        delete_empty_lines=True,
        remove_special_chars=True,
    ) -> Dataset | DatasetDict:
        dataset = self.dataset[subset_name]
        dataset = dataset.add_column(
            "_corresponded_docs", self.qa_corresponding_prompts[subset_name]
        )

        if remove_special_chars:
            dataset = dataset.map(
                self._remove_special_chars_map_fn,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                desc=f"Removing special characters in {subset_name} dataset",
            )

        if delete_empty_lines:
            dataset = dataset.filter(
                lambda example: len(example[self.answer_column].strip()) != 0
            )

        column_names = dataset.column_names
        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            desc=f"Running tokenizer on {subset_name} dataset",
        )
        column_names.append("text_inputs")
        dataset.set_format(
            type="torch",
            columns=list(set(dataset.column_names).difference(set(column_names))),
        )

        return dataset

    def _preprocess_function(self, examples):
        inputs, targets = examples[self.question_column], examples[self.answer_column]
        inputs = [
            self.prompt_template.format(doc, question)
            for doc, question in zip(examples["_corresponded_docs"], inputs)
        ]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )
        # Tokenize targets with text_target=...
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_answer_length,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id
        # in the labels by ignore_pad_token when we want to ignore
        # padding in the loss.

        self._pad_input_ids(labels)

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["text_inputs"] = inputs
        return model_inputs


class LMQADataPreprocessor(DataPreprocessorBase):
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        question_column: str,
        answer_column: str,
        preprocessing_num_workers: int = 32,
        start_text_template: str = "Question: {}\nAnswer: ",
        end_text_template: str = "{}",
    ):
        super().__init__(
            dataset,
            tokenizer,
            question_column,
            answer_column,
            preprocessing_num_workers,
        )

        self.start_text_template = start_text_template
        self.end_text_template = end_text_template

    def _preprocess_function(self, examples):
        questions, answers = (
            examples[self.question_column],
            examples[self.answer_column],
        )
        questions = [q.replace("\n", " ").replace("\r", " ") for q in questions]
        answers = [a.replace("\n", " ").replace("\r", " ") for a in answers]

        start_text = [
            self.start_text_template.format(question) for question in questions
        ]
        full_text = [
            self.start_text_template.format(question)
            + self.end_text_template.format(answer)
            + self.tokenizer.eos_token
            for question, answer in zip(questions, answers)
        ]

        return {
            "full_text": full_text,
            "start_text": start_text,
        }
