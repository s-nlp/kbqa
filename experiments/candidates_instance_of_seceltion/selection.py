#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import pickle
from tqdm.auto import tqdm
import time

from question_to_rank import Entity, QuestionToRank
from kbqa.logger import get_logger


# In[9]:


logger = get_logger()


# In[2]:


seq2seq_results_df = pd.read_csv(
    "../../../runs/wdsq_tunned/google_t5-large-ssm/evaluation/version_4/results.csv"
)
seq2seq_results_df.head()

answers_cols = seq2seq_results_df.columns[2:-1]
answers_cols

with open("../simplequestions_candidate_selection_dataset/test.pkl", "rb") as fh:
    test_df = pickle.load(fh)

seq2seq_results_df["question"] = seq2seq_results_df["question"].apply(
    lambda s: s.replace("\n", "")
)

final_test_df = seq2seq_results_df.merge(test_df, left_on="question", right_on="Q")
final_test_df.head()


# In[10]:


for _, row in tqdm(final_test_df.iterrows(), total=final_test_df.index.size):
    start_time = time.time()
    question_entities = [Entity(idx) for idx in row["one_hop_neighbors"].keys()]

    answers_candidates = []
    for label in row[answers_cols].unique():
        try:
            answers_candidates.append(Entity.from_label(label)[0])
        except ValueError:
            pass

    qtr = QuestionToRank(
        row["question"],
        question_entities,
        answers_candidates,
    )
    qtr.final_answers()

    logger.info(
        {
            "msg": f"Question {row['question']} processed",
            "elapsed_time[ms]": time.time() - start_time,
        }
    )
    print("elapsed_time[ms]", time.time() - start_time)

# In[ ]:
