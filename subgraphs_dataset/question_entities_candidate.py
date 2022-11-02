"""
Module to help organize the question, entities, candidates and scores.
Essentially is an object that hold our wanted info
"""
import numpy as np


class QuestionEntitiesCandidates:
    """class that holds the questions, entities and candidates"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, original_question: str, ner_question: str) -> None:
        self.original_question = original_question
        self.ner_question = ner_question
        self.entity_texts = []
        self.entity_ids = []
        self.entity_scores = []
        self.candidate_texts = []
        self.candidate_ids = []
        self.original_entities = None

    def get_entities(self, dirty_entities, lang):
        """
        return the entity texts from generated_entities_batched
        based on the language.

        format of diry entities:
        [
            {'id': 'some_id',
            'texts': ['some text >> en', 'some text >> br'],
            'scores': tensor([score array]),
            'score': tensor(score value average from scores)}
            ,...
        ]
        """
        # in case we need the scores in the future
        self.original_entities = dirty_entities

        for entity in dirty_entities:
            entity_id = entity["id"]
            entity_text = entity["texts"]
            entity_score = entity["score"]
            entity_text_lang = None

            # getting our entity in the wanted language
            if len(entity_text) > 1:  # received more than one languages
                for text in entity_text:
                    text = text.split(">>")
                    if text[-1].strip() == lang:
                        entity_text_lang = text[0].strip()
            else:  # only 1 lang -> english
                text = entity_text[0].split(">>")
                if text[-1].strip() == lang:
                    entity_text_lang = text[0].strip()
            if entity_text_lang is not None:
                self.entity_texts.append(entity_text_lang)
            else:
                self.entity_texts.append(None)

            self.entity_scores.append(float(entity_score))
            self.entity_ids.append(entity_id)

    def clean_candidates(self, candidates, num_ans):
        """
        clean the candidates and get num_ans wrong answers
        """
        target_answer = candidates[0]
        res = [target_answer]
        curr_wrong_ans = 0
        uniq_candidates = np.unique(candidates[1:]).tolist()

        for candidate in uniq_candidates:
            if isinstance(candidate, str):
                candidates = candidate.strip()

                # getting num_ans wrong answers
                if candidate != target_answer and curr_wrong_ans < num_ans:
                    res.append(candidate)
                    curr_wrong_ans += 1

                    if curr_wrong_ans == num_ans:
                        break

        return res

    def populate_candidates(self, candidate_texts, label2entity, num_ans):
        """
        given the list of natural language candidates, set both texts
        and id's in our class
        """
        # clean the candidates
        candidate_texts = self.clean_candidates(candidate_texts, num_ans)
        self.candidate_texts = candidate_texts

        # get the candidate id
        for candidate in candidate_texts:
            curr_id = label2entity.get_id(candidate)
            self.candidate_ids.append(curr_id)

    def display(self):
        """
        print out our current question, entity id, score, and candidate
        """
        print("original question:", self.original_question)
        print("entities:", self.entity_ids)
        print("candidates:", self.candidate_ids)
