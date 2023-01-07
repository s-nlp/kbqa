#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

import torch

from kbqa.wikidata import Entity


# In[2]:


train_df = pd.read_csv('./wdsq_dataset/train_no_prefix_tree.csv')
valid_df = pd.read_csv('./wdsq_dataset/valid_no_prefix_tree.csv') 

answer_cols = [c for c in train_df.columns if 'answer_' in c]


# In[3]:


def _proc_row(row):
    candidates = [e for _, e in Entity(row['S']).forward_one_hop_neighbors]
    for lbl in row[answer_cols].unique():
        try:
            candidates.append(Entity.from_label(str(lbl))[0])
        except ValueError:
            pass
    candidates = list(set(candidates))
    
    target = Entity(row['O'])

    items = []
    for candidate in candidates:
        candidate.label
        
        item = {}
        item['question'] = row['Q']
        item['answer'] = candidate.idx
        item['target'] = target.idx
        item['is_correct'] = candidate == target
        items.append(item)
    return items


# In[4]:


# train_dataset = Parallel(n_jobs=6)(
#     delayed(_proc_row)(row)
#     for _, row in tqdm(train_df.iterrows(), total=train_df.index.size)
# )

valid_dataset = Parallel(n_jobs=6)(
    delayed(_proc_row)(row)
    for _, row in tqdm(valid_df.iterrows(), total=valid_df.index.size)
)


# In[ ]:




