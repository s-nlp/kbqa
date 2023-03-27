from kbqa.candidate_selection import QuestionCandidateRelationSelection
from fastapi import FastAPI
from typing import List

app = FastAPI()


@app.post("/relation_selection/one_hop_direct_connections/")
def one_hop_direct_connections(
    question_entities_ids: List[str], candidates_ids: List[str]
):
    return QuestionCandidateRelationSelection.filter_one_hop_direct_connections(
        question_entities_ids, candidates_ids
    )


@app.post("/relation_selection/two_hop_direct_connections/")
def two_hop_direct_connections(
    question_entities_ids: List[str], candidates_ids: List[str]
):
    return QuestionCandidateRelationSelection.filter_two_hop_direct_connections(
        question_entities_ids, candidates_ids
    )
