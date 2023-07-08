"""
Module to help organize the question, entities, candidates and scores.
Essentially is an object that hold our wanted info
"""


class QuestionEntitiesCandidates:
    """class that holds the questions, entities and candidates"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, original_question: str) -> None:
        self.original_question = original_question
        self.entity_texts = []
        self.entity_ids = []
        self.entity_scores = []
        self.candidate_texts = []
        self.candidate_ids = []
        self.original_entities = None

    def populate_candidates(self, candidates, entity2label, num_ans):
        """
        given the list of natural language candidates, set both texts
        and id's in our class
        """
        curr_wrong_ans = 0
        gold_candidate = candidates[0]
        gold_candidate_txt = entity2label.get_label(gold_candidate)
        self.candidate_texts.append(gold_candidate_txt)
        self.candidate_ids.append(gold_candidate)

        for candidate in candidates[1:]:
            if candidate not in self.candidate_ids:
                candidate_txt = entity2label.get_label(candidate)
                self.candidate_texts.append(candidate_txt)
                self.candidate_ids.append(candidate)

                curr_wrong_ans += 1
                if curr_wrong_ans == num_ans + 1:
                    break

    def display(self):
        """
        print out our current question, entity id, score, and candidate
        """
        print()
        print("original question:", self.original_question)
        print("entities:", self.entity_ids)
        print("candidates:", self.candidate_ids)
