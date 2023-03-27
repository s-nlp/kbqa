import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import pandas as pd
import torch
from nltk.stem.porter import PorterStemmer
from pywikidata import Entity
from seq2seq_dbs_answers_generation import load_datasets, load_params
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline

from kbqa.caches.ner_to_sentence_insertion import NerToSentenceInsertion


class MGENREPipeline(Pipeline):
    """MGENREPipeline - HF Pipeline for mGENRE EntityLinking model"""

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        if "num_beams" in kwargs:
            forward_kwargs["num_beams"] = kwargs.get("num_beams", 20)
        if "num_return_sequences" in kwargs:
            forward_kwargs["num_return_sequences"] = kwargs.get(
                "num_return_sequences", 20
            )
        if "mgenre_trie" in kwargs:
            forward_kwargs["mgenre_trie"] = kwargs.get("mgenre_trie")
        return {}, forward_kwargs, {}

    def preprocess(self, input_):
        return self.tokenizer(
            input_,
            return_tensors="pt",
        )

    def _forward(
        self,
        input_tensors,
        num_beams=10,
        num_return_sequences=10,
        mgenre_trie=None,
    ):
        outputs = self.model.generate(
            **{k: v.to(self.device) for k, v in input_tensors.items()},
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            prefix_allowed_tokens_fn=lambda batch_id, sent: mgenre_trie.get(
                sent.tolist()
            )
            if mgenre_trie is not None
            else None,
        )
        return outputs

    def postprocess(self, model_outputs):
        outputs = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs


class EntitiesSelection:
    def __init__(self, ner_model):
        self.stemmer = PorterStemmer()
        self.ner_model = ner_model

    def entities_selection(self, entities_list, mgenre_predicted_entities_list):
        final_preds = []

        for pred_text in mgenre_predicted_entities_list:
            labels = []
            try:
                _label, lang = pred_text.split(" >> ")
                if lang == "en":
                    labels.append(_label)
            except Exception as e:
                print(f"Error {str(e)} with pred_text={pred_text}")

            if len(labels) > 0:
                for label in labels:
                    label = label.lower()
                    if self._check_label_fn(label, entities_list):
                        final_preds.append(pred_text)

        return list(dict.fromkeys(final_preds))

    @lru_cache(maxsize=8192)
    def _label_format_fn(self, label):
        return " ".join(
            [self.stemmer.stem(str(token)) for token in self.ner_model(label)]
        )

    def _check_label_fn(self, label, entities_list):
        label = self._label_format_fn(label)
        for entity in entities_list:
            entity = self._label_format_fn(entity)
            if label == entity:
                return True
        return False


if __name__ == "__main__":
    params, run_name = load_params()
    train_dataset, valid_dataset, test_dataset, question_col_name = load_datasets(
        params
    )
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    mgenre_tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
    mgenre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()

    mgenre = MGENREPipeline(
        mgenre_model,
        mgenre_tokenizer,
        device=device,
    )

    ner = NerToSentenceInsertion(model_path=params["entity_linking"]["ner"]["path"])
    entities_selection_module = EntitiesSelection(ner.model)

    for split_name, dataset in [
        ("train", train_dataset),
        ("valid", valid_dataset),
        ("test", test_dataset),
    ]:
        if dataset is not None:
            final_results = defaultdict(list)
            for q in tqdm(dataset[question_col_name]):
                text_with_labeling, entities_list = ner.entity_labeling(q, True)
                mgenre_results = mgenre(text_with_labeling)
                selected_mgenre_results = entities_selection_module.entities_selection(
                    entities_list, mgenre_results
                )

                entities = []
                for lbl in selected_mgenre_results:
                    try:
                        lbl = lbl.split(" >> ")[0]
                    except:
                        lbl = lbl[: lbl.index(" >>")]
                        lbl = lbl[: lbl.index(">>")]
                    try:
                        entities.extend([e.idx for e in Entity.from_label(lbl)])
                    except ValueError:
                        pass

                final_results["question"].append(q)
                final_results["mgenre_results"].append(mgenre_results)
                final_results["selected_mgenre_results"].append(selected_mgenre_results)
                final_results["selected_entities"].append(entities)

            Path(
                f"/mnt/raid/data/kbqa/datasets/linked_entities/{run_name}/{params['seq2seq']['dataset']}/"
            ).mkdir(parents=True, exist_ok=True)
            final_results_df = pd.DataFrame(final_results)
            final_results_df.to_pickle(
                f"/mnt/raid/data/kbqa/datasets/linked_entities/{run_name}/{params['seq2seq']['dataset']}/{split_name}.pkl"
            )
