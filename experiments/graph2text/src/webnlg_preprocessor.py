from __future__ import annotations

import copy
import random
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.preprocessor_base import DataPreprocessorBase


class WebNLGPreprocessor(DataPreprocessorBase):
    def __init__(
        self,
        dataset: Dataset | DatasetDict | List[Dict],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        sequence_in: str,
        sequence_out: str,
        max_seq_length: int,
        max_answer_length: int,
        padding: str,
        start_text_template: str = "",
        ignore_pad_token_for_loss: bool = True,
        ignore_pad_token=-100,
        preprocessing_num_workers: int = 32,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            sequence_in=sequence_in,
            sequence_out=sequence_out,
            preprocessing_num_workers=preprocessing_num_workers,
        )
        self.max_input_length = max_seq_length
        self.max_output_length = max_answer_length
        self.start_text_template = start_text_template

        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.ignore_pad_token = ignore_pad_token

        print("Total samples = {}".format(len(self.dataset)))

        self.metric = "BLEU"

        self.head_ids, self.rel_ids, self.tail_ids = (
            self.tokenizer.encode(" [head]", add_special_tokens=False),
            self.tokenizer.encode(" [relation]", add_special_tokens=False),
            self.tokenizer.encode(" [tail]", add_special_tokens=False),
        )

        self.graph_ids, self.text_ids = self.tokenizer.encode(
            " [graph]", add_special_tokens=False
        ), self.tokenizer.encode(" [text]", add_special_tokens=False)

        if self.tokenizer.bos_token_id:
            self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

        if start_text_template == "":
            self.add_bos_id_for_quest = self.add_bos_id
        else:
            self.add_bos_id_for_quest = (
                self.add_bos_id
                + self.tokenizer(start_text_template, add_special_tokens=False)[
                    "input_ids"
                ]
            )

    def get_preprocessed_dataset(
        self,
        subset_name="train",
        delete_empty_lines=True,
        remove_initial_columns=True,
        remove_special_chars=True,
    ) -> Dataset | DatasetDict:
        dataset = self.dataset[subset_name]
        # dataset = dataset[:100]
        output_dict = [self.__getitem__(data) for data in dataset]
        dataset_dict = Dataset.from_list(output_dict)

        return dataset_dict

    def linearize_v2(
        self, entity, entity_change, head_ids, rel_ids, tail_ids, relation_change
    ):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], "", [], []

        string_label = copy.deepcopy(head_ids)
        string_label_tokens = " [head]"

        string_label += entity_change[entity[0]][0]
        string_label_tokens += " {}".format(entity[0])

        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0]
                words_label_tokens = " [relation] {} [tail] {}".format(
                    rel_label_token, rel[1]
                )

                string_label += words_label
                string_label_tokens += words_label_tokens

        return string_label, string_label_tokens

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry["kbs"][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity) + list(text_relation)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(
                " {}".format(total_entity[ent_id]), add_special_tokens=False
            )
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]
        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(
                " {}".format(text_relation[rel_id]), add_special_tokens=False
            )
        return ent_change, rel_change

    def truncate_pair_ar(self, a, graph_ids, text_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = (
            self.max_input_length
            - len(self.add_bos_id_for_quest)
            - len(graph_ids)
            - len(text_ids)
            - 1
        )
        if len(a) > length_a_b:
            a = a[:length_a_b]
        input_ids = (
            self.add_bos_id_for_quest
            + graph_ids
            + a
            + text_ids
            + [self.tokenizer.eos_token_id]
        )

        attn_mask = [1] * len(input_ids) + [0] * (
            self.max_input_length - len(input_ids)
        )
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_input_length - len(input_ids)
        )

        return input_ids, attn_mask

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[
                : (self.max_output_length - len(add_bos_id) - 1)
            ]
        decoder_label_ids = (
            add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        )
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (
            self.max_output_length - len(decoder_label_ids)
        )
        decoder_label_ids += [self.tokenizer.pad_token_id] * (
            self.max_output_length - len(decoder_label_ids)
        )
        assert (
            len(decoder_label_ids) == self.max_output_length == len(decoder_attn_mask)
        )

        input_ids, input_attn_mask = self.truncate_pair_ar(
            questions, graph_ids, text_ids
        )

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask

    def _preprocess_function(self, examples):
        output_ids = {"input_ids": [], "labels": []}
        for example in examples:
            output_id = self.__getitem__(example)
            output_ids["input_ids"].append(output_id["input_ids"])
            output_ids["labels"].append(output_id["labels"])

        return output_ids

    def __getitem__(self, example):
        entry = example
        entities = []
        for _ in entry["kbs"]:
            entities.append(_)

        strings_label = []
        strings_label_tokens = ""

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry["kbs"][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(
            mark_entity_number, mark_entity, entry
        )
        entity_change, relation_change = self.get_change_per_sample(
            mark_entity, text_entity, text_relation
        )

        for i, entity_id in enumerate(entities):
            entity = entry["kbs"][entity_id]
            string_label, string_label_tokens = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids,
                self.tail_ids,
                relation_change,
            )

            strings_label += string_label
            strings_label_tokens += string_label_tokens

        words_label_ids, words_label_tokens = (
            [],
            "",
        )
        current_text = random.choice(entry["text"])

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(
                " {}".format(word), add_special_tokens=False
            )
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += " " + word_label_tokens

        (
            input_ids_ar,
            attn_mask_ar,
            decoder_label_ids,
            decoder_attn_mask,
        ) = self.ar_prep_data(
            words_label_ids,
            strings_label,
            self.add_bos_id,
            self.graph_ids,
            self.text_ids,
        )

        assert len(input_ids_ar) == len(attn_mask_ar) == self.max_input_length
        assert (
            len(decoder_label_ids) == len(decoder_attn_mask) == self.max_output_length
        )

        input_ids_ar = torch.LongTensor(input_ids_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)

        out_dict = {}
        out_dict["input_ids"] = input_ids_ar
        out_dict["labels"] = decoder_label_ids

        return out_dict
