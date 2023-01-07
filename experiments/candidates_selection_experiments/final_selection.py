#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
from tqdm.auto import tqdm
import time

from joblib import Parallel, delayed

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from kbqa.wikidata import Entity
from kbqa.candidate_selection import QuestionToRankInstanceOf
from kbqa.logger import get_logger


# In[3]:


logger = get_logger()

answers_df = pd.read_csv('./filtered_test_with_answers.csv')
mgenre_df = pd.read_pickle('./filtered_test_with_mgenre_no_prefix_tree.pkl')
# answers_df = pd.read_csv('rubq_test_with_answers_no_prefix.csv')
# mgenre_df = pd.read_pickle('./test_rubq2_mgenre.pkl')

mgenre_df = mgenre_df.loc[:,~mgenre_df.columns.duplicated()]
answers_df = answers_df.loc[:,~answers_df.columns.duplicated()]

mgenre_df.head()


# In[4]:


answer_cols = [c for c in answers_df.columns if 'answer_' in c]

df = pd.merge(left=answers_df[['Q']+answer_cols], right=mgenre_df[['S', 'O', 'Q', 'selected_entities']], left_on='Q', right_on='Q', how='left')[['S', 'O', 'Q', 'selected_entities']+answer_cols]
df.head()


# In[5]:


def _row_proc(row):
    try:
        answers_candidates = []
        for lbl in row[answer_cols].dropna().unique():
            try:
                answers_candidates.extend(Entity.from_label(lbl)[:2])
            except ValueError:
                pass

        # question_entities = [Entity(e) for e in row['selected_entities']]
        question_entities = [Entity(row['S'])]

        qtr = QuestionToRankInstanceOf(
            row['Q'],
            question_entities,
            answers_candidates,
            only_forward_one_hop=False,
        )

        answers = qtr.final_answers()
        if len(answers) > 0:
            return answers[0][1].idx
        else:
            return None
    except Exception as e:
        print(e)

filtered_answers = Parallel(n_jobs=6)(
    delayed(_row_proc)(row)
    for _, row in tqdm(df.iterrows(), total=df.index.size)
)

# In[5]:


df['filtered_answer'] = filtered_answers
df['filtered_answer'] = df['filtered_answer'].apply(Entity)

df['is_correct'] = df.apply(
    lambda row: Entity(row['O']) == row['filtered_answer'],
    axis=1
)

df[df['is_correct']].index.size / df.index.size


# 0.42945036915504514

# In[ ]:





# In[6]:


class QuestionToRankInstanceOfHtml(QuestionToRankInstanceOf):
    def _repr_html_(self) -> str:
        html = ["""
        <style>
        .flex-row-container {
            display: flex;
            flex-wrap: wrap;
        }
        .flex-row-container > .flex-row-item {
            flex: 1 0 29%; /*grow | shrink | basis */
        }

        .flex-row-item {
            margin: 10px;
        }

        th {
            word-wrap: break-word;
        }

        td {
            word-wrap: break-word;
        }
        </style>
        """]
        html.append(f'<b>Question:</b> {self.question}')
        if self.target is not None:
            html.append(
                f'<br><b>Target:</b> <span style="color: #3CF30F">Entity: {self.target.idx} ({self.target.label})</span> (InstanceOf: {"; ".join([f"{e.idx} ({e.label})" for e in self.target.instance_of])})'
            )
        html.append('<div class="flex-row-container">')

        # FINAL ANSWERS
        html_final_answers = ["<h4>Final answers</h4>"]
        html_final_answers.extend([
            '<table style="border: 1px solid #A9ED2B">',
            '<tr style="font-size:1rem; font-weight: bold; background-color: #A9ED2B">',
            "<th>Entity</th>",
            "<th>E Label</th>",
            "<th>InstanceOf</th>",
            "<th>forward one hop neighbors score</th>",
            "<th>answers candidates score</th>",
            "<th>property question intersection score</th>",
            "</tr>"
        ])
        for property, entity, forward_one_hop_neighbors_score, answers_candidates_score, property_question_intersection_score in self.final_answers():
            if self.target is not None and self.target == entity:
                html_final_answers.append('<tr style="background-color: #3CF30F">')
            else:
                html_final_answers.append('<tr>')
            html_final_answers.extend([
                f'<td>{entity.idx}</td>',
                f'<td>{entity.label}</td>',
                f'<td>{"<br>".join([f"{io.idx} ({io.label})" for io in entity.instance_of])}</td>',
                f'<td>{forward_one_hop_neighbors_score}</td>',
                f'<td>{answers_candidates_score}</td>',
                f'<td>{property_question_intersection_score}</td>',
                '</tr>',
            ])
        html_final_answers.append("</table>")
        html_final_answers = "".join(html_final_answers)
        
        # QUESTION
        html_question_entities = []
        for qentity in self.question_entities:
            html_question_entities.append(f"<h4>One hop neighbors for Entity: {qentity.idx} ({qentity.label})</h4>")
            html_question_entities.extend([
                '<table style="width: 600px;">',
                '<tr style="font-size:1rem; font-weight: bold; background-color: #50ADFF">',
                '<th>Dir</th>',
                "<th>Property</th>",
                "<th>P Label</th>",
                "<th>Entity</th>",
                "<th>E Label</th>",
                "<th>InstanceOf</th>",
                "</tr>"
            ])

            if self.only_forward_one_hop:
                neighbors = qentity.forward_one_hop_neighbors
            else:
                neighbors = qentity.forward_one_hop_neighbors + qentity.backward_one_hop_neighbors

            for property, entity in neighbors:
                if self.target is not None and self.target == entity:
                    html_question_entities.append('<tr style="background-color: #3CF30F">')
                elif entity in self.final_answers():
                    html_question_entities.append('<tr style="background-color: #A9ED2B">')
                else:
                    html_question_entities.append('<tr>')
                html_question_entities.extend([
                    f'<td>{"->" if (property, entity) in qentity.forward_one_hop_neighbors else "<-"}</td>'
                    f'<td>{property.idx}</td>',
                    f'<td>{property.label}</td>',
                    f'<td>{entity.idx}</td>',
                    f'<td>{entity.label}</td>',
                    f'<td>{"<br>".join([f"{io.idx} ({io.label})" for io in entity.instance_of])}</td>',
                    '</tr>',
                ])

            html_question_entities.append('</table>')
        
        html_question_entities = '<div class="flex-row-item">' + html_final_answers + "".join(html_question_entities) + '</div>'
        html.append(html_question_entities)

        # ANSWERS_INSTANCE_OF_COUNT
        html_answer_instance_of = ""
        html_answer_instance_of = [
            '<h4>Answers instanceOf count (<b style="color: green;">selected</b>)</h4>',
            "<table>",
            '<tr style="font-size:1rem; font-weight: bold; background-color: #50ADFF">',
            "<th>InstanceOf</th>",
            "<th>Label</th>",
            "<th>Count</th>",
            "</tr>"
        ]
        for instance_of_entity, count in self.answer_instance_of_count:
            if instance_of_entity in self._answer_instance_of:
                html_answer_instance_of.append('<tr style="background-color: #7AE2BC">')
            else:
                html_answer_instance_of.append('<tr>')
            html_answer_instance_of.append(f'<td>{instance_of_entity.idx}</td>')
            html_answer_instance_of.append(f'<td>{instance_of_entity.label}</td>')
            html_answer_instance_of.append(f'<td>{count}</td>')
            html_answer_instance_of.append('</tr>')  
        html_answer_instance_of.append("</table>")

        html_answer_instance_of = "".join(html_answer_instance_of)

        # ANSWERS candidates
        html_answers_candidates = [f"<h4>Seq2Seq answers candidates</h4>"]
        html_answers_candidates.extend([
            "<table>",
            '<tr style="font-size:1rem; font-weight: bold; background-color: #50ADFF">',
            "<th>Entity</th>",
            "<th>E Label</th>",
            "<th>InstanceOf</th>",
            "</tr>"
        ])
        for entity in self.answers_candidates:
            if self.target is not None and self.target == entity:
                html_answers_candidates.append('<tr style="background-color: #3CF30F">')
            else:
                html_answers_candidates.append('<tr>')
            html_answers_candidates.extend([
                f'<td>{entity.idx}</td>',
                f'<td>{entity.label}</td>',
                f'<td>{"<br>".join([f"{io.idx} ({io.label})" for io in entity.instance_of])}</td>',
                '</tr>',
            ])

        html_answers_candidates = '<div class="flex-row-item">' + html_answer_instance_of + "".join(html_answers_candidates) + '</div>'
        html.append(html_answers_candidates)

        return "".join(html) + '</div>'



for i in range(100):
    row = df[~df['is_correct']].iloc[i]
    answers_candidates = []
    for lbl in row[answer_cols].dropna().unique():
        try:
            answers_candidates.extend(Entity.from_label(lbl)[:1])
        except ValueError:
            pass
    question_entities = [Entity(e) for e in row['selected_entities']]

    qtr = QuestionToRankInstanceOfHtml(
        row['Q'],
        question_entities,
        answers_candidates,
        target_entity=Entity(row['O']),
        only_forward_one_hop=False,
    )

    qtr.final_answers()
    qtr._repr_html_()


# In[ ]:




