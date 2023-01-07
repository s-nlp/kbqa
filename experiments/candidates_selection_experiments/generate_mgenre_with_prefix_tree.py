#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import pandas as pd
import datasets
from pathlib import Path
from transformers import Pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional, Dict
import pickle
import torch
from torch.utils.data import DataLoader
from functools import lru_cache
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

tqdm.pandas()

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from kbqa.wikidata import Entity
from kbqa.utils.train_eval import get_best_checkpoint_path
from kbqa.caches.ner_to_sentence_insertion import NerToSentenceInsertion

from trie import MarisaTrie


# In[2]:


# get_ipython().system('git clone https://github.com/askplatypus/wikidata-simplequestions.git')

test_df = pd.read_csv(
    "./wikidata-simplequestions/annotated_wd_data_test_answerable.txt",
    sep="\t",
    names=["S", "P", "O", "Q"],
)

train_df = pd.read_csv(
    "./wikidata-simplequestions/annotated_wd_data_train_answerable.txt",
    sep="\t",
    names=["S", "P", "O", "Q"],
)

valid_df = pd.read_csv(
    "./wikidata-simplequestions/annotated_wd_data_valid_answerable.txt",
    sep="\t",
    names=["S", "P", "O", "Q"],
)


# In[3]:


# import numpy as np

# filtered_test = np.load('simple_questions_filtered.npy')
# test_df = pd.DataFrame(filtered_test, columns=["S", "P", "O", "Q"])
# test_df.head()


# In[4]:


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
            prefix_allowed_tokens_fn=
                lambda batch_id, sent: mgenre_trie.get(
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


# In[5]:


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

mgenre_tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
mgenre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()

# with open('./wdsq_mgenre_trie.pkl', 'rb') as f:
#     trie = pickle.load(f)

mgenre = MGENREPipeline(
    mgenre_model,
    mgenre_tokenizer,
    # mgenre_trie=trie,
    device=device,
)


ner = NerToSentenceInsertion(model_path='../../../ner/')
entities_selection_module = EntitiesSelection(ner.model)


# In[7]:


df = train_df

final_results = defaultdict(list)

for q in tqdm(df['Q'].values):
    text_with_labeling, entities_list = ner.entity_labeling(q, True)
    mgenre_results = mgenre(text_with_labeling)
    selected_mgenre_results = entities_selection_module.entities_selection(entities_list, mgenre_results)

    entities = []
    for lbl in selected_mgenre_results:
        try:
            lbl = lbl.split(' >> ')[0]
        except:
            lbl = lbl[:lbl.index(' >>')]
            lbl = lbl[:lbl.index('>>')]
        try:
            entities.extend(Entity.from_label(lbl))
        except ValueError:
            pass
    
    final_results['Q'].append(q)
    final_results['mgenre_results'].append(mgenre_results)
    final_results['selected_mgenre_results'].append(selected_mgenre_results)
    final_results['selected_entities'].append(entities)


final_df = pd.concat([df, pd.DataFrame(final_results)], axis=1)
final_df['selected_entities'] = final_df['selected_entities'].apply(lambda x: [_x.idx for _x in x])

final_df.to_pickle('wdsq_train_with_mgenre_no_prefix_tree.pkl')
final_df.to_csv('wdsq_train_with_mgenre_no_prefix_tree.csv', index=False)





df = valid_df

final_results = defaultdict(list)

for q in tqdm(df['Q'].values):
    text_with_labeling, entities_list = ner.entity_labeling(q, True)
    mgenre_results = mgenre(text_with_labeling)
    selected_mgenre_results = entities_selection_module.entities_selection(entities_list, mgenre_results)

    entities = []
    for lbl in selected_mgenre_results:
        try:
            lbl = lbl.split(' >> ')[0]
        except:
            lbl = lbl[:lbl.index(' >>')]
            lbl = lbl[:lbl.index('>>')]
        try:
            entities.extend(Entity.from_label(lbl))
        except ValueError:
            pass
    
    final_results['Q'].append(q)
    final_results['mgenre_results'].append(mgenre_results)
    final_results['selected_mgenre_results'].append(selected_mgenre_results)
    final_results['selected_entities'].append(entities)


final_df = pd.concat([df, pd.DataFrame(final_results)], axis=1)
final_df['selected_entities'] = final_df['selected_entities'].apply(lambda x: [_x.idx for _x in x])

final_df.to_pickle('wdsq_valid_with_mgenre_no_prefix_tree.pkl')
final_df.to_csv('wdsq_valid_with_mgenre_no_prefix_tree.csv', index=False)







df = test_df

final_results = defaultdict(list)

for q in tqdm(df['Q'].values):
    text_with_labeling, entities_list = ner.entity_labeling(q, True)
    mgenre_results = mgenre(text_with_labeling)
    selected_mgenre_results = entities_selection_module.entities_selection(entities_list, mgenre_results)

    entities = []
    for lbl in selected_mgenre_results:
        try:
            lbl = lbl.split(' >> ')[0]
        except:
            lbl = lbl[:lbl.index(' >>')]
            lbl = lbl[:lbl.index('>>')]
        try:
            entities.extend(Entity.from_label(lbl))
        except ValueError:
            pass
    
    final_results['Q'].append(q)
    final_results['mgenre_results'].append(mgenre_results)
    final_results['selected_mgenre_results'].append(selected_mgenre_results)
    final_results['selected_entities'].append(entities)


final_df = pd.concat([df, pd.DataFrame(final_results)], axis=1)
final_df['selected_entities'] = final_df['selected_entities'].apply(lambda x: [_x.idx for _x in x])

final_df.to_pickle('wdsq_test_with_mgenre_no_prefix_tree.pkl')
final_df.to_csv('wdsq_test_with_mgenre_no_prefix_tree.csv', index=False)
