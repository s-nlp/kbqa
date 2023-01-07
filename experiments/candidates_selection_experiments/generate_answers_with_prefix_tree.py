#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import pandas as pd
import datasets
from pathlib import Path
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import List, Optional, Dict
import pickle
import torch
from torch.utils.data import DataLoader

tqdm.pandas()

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from kbqa.wikidata import Entity
from kbqa.utils.train_eval import get_best_checkpoint_path

from trie import MarisaTrie


# In[2]:


# get_ipython().system('nvidia-smi')


# In[3]:


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


# In[4]:


import numpy as np

filtered_test = np.load('simple_questions_filtered.npy')
test_df = pd.DataFrame(filtered_test, columns=["S", "P", "O", "Q"])
print(test_df.index.size)
test_df.head()


# In[5]:


tokenizer = AutoTokenizer.from_pretrained("google/t5-large-ssm")

def convert_to_features(example_batch):
    input_encodings = tokenizer(
        example_batch['Q'],
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    return input_encodings

model = T5ForConditionalGeneration.from_pretrained(
    get_best_checkpoint_path(Path('../../../runs/wdsq_tunned/google_t5-large-ssm/models/'))
)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_return_sequences=200
num_beams=200
num_beam_groups=20
diversity_penalty=0.1
batch_size=2


# In[6]:


# get_ipython().system('mkdir ./wdsq_dataset/')

def prepare_dataset(df, name):
    test_dataset = datasets.Dataset.from_pandas(df)
    test_dataset = test_dataset.map(convert_to_features, batched=True)
    test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    generated_decoded = {f"answer_{idx}": [] for idx in range(num_return_sequences)}
    for batch in tqdm(dataloader, desc="evaluate model"):
        generated_ids = model.generate(
            batch["input_ids"].to(device),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens,
        )
        generated_decoded_batch = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        current_batch_size = batch["input_ids"].shape[0]
        for start_batch_idx in range(
            0, current_batch_size * num_return_sequences, num_return_sequences
        ):
            for answer_idx, answer in enumerate(
                generated_decoded_batch[
                    start_batch_idx : start_batch_idx + num_return_sequences
                ]
            ):
                generated_decoded[f"answer_{answer_idx}"].append(answer)

    _df = pd.concat([df, pd.DataFrame(generated_decoded)], axis=1)
    _df.to_csv(f'./wdsq_dataset/{name}_no_prefix_tree.csv', index=False)
    _df.to_pickle(f'./wdsq_dataset/{name}_no_prefix_tree.pkl')

prepare_dataset(train_df, 'train')
prepare_dataset(valid_df, 'valid')
prepare_dataset(test_df, 'filtered_test')
