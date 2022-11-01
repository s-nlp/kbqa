"""Evaluates top-n and top-1 accuracies on WDSQ and RuBQ 1-hop"""

import pickle
import numpy as np

MASTER_SEED = 42


def mgenre_single_eval():
    """Evaluates top-n and top-1 accuracies on WDSQ and RuBQ 1-hop"""
    rubq_entities = np.load("rubq_test_entities.npy")
    sq_test = np.load("./data/simple_questions_test.npy")
    sq_mask = np.load("./data/sq_mask_present_in_graph.npy")

    old_rubq_candidates = np.load(
        "data/presearched_fixed_rubq_test.npy", allow_pickle=True
    )
    old_rubq_candidates = [list(q.keys()) for q in old_rubq_candidates]

    old_sq_candidates = np.load(
        "data/candidate_entities_sq_test.npy", allow_pickle=True
    )
    old_sq_candidates = [list(q.item()) for q in old_sq_candidates]

    sq_entities = []
    sq_questions = []
    for (entity, _, _, question) in sq_test:
        sq_entities.append(entity)
        sq_questions.append(question)
    masked_sq_entities = np.array(sq_entities)[sq_mask]

    sq_results = None
    rubq_results = None

    with open(f"results_sq_bs_20_stanza.pickle", "rb") as handle:
        sq_results = pickle.load(handle)

    with open(f"results_rubq_bs_20_stanza.pickle", "rb") as handle:
        rubq_results = pickle.load(handle)

    sq_candidates = []
    rubq_candidates = []

    for sq_result in sq_results:
        sq_candidates.append([result["id"] for result in sq_result])

    for rubq_result in rubq_results:
        rubq_candidates.append([result["id"] for result in rubq_result])

    masked_sq_candidates = np.array(sq_candidates)[sq_mask]

    rubq_union_candidates = []
    for (mgenre_candidates, old_candidates) in zip(
        rubq_candidates, old_rubq_candidates
    ):
        rubq_union_candidates.append(list(set(mgenre_candidates + old_candidates)))

    sq_union_candidates = []
    for (mgenre_candidates, old_candidates) in zip(
        masked_sq_candidates, old_sq_candidates
    ):
        sq_union_candidates.append(list(set(mgenre_candidates + old_candidates)))

    sq_entity_found = 0
    for candidates, entity in zip(sq_candidates, sq_entities):
        if entity in candidates:
            sq_entity_found += 1

    masked_sq_entity_found = 0
    for candidates, entity in zip(masked_sq_candidates, masked_sq_entities):
        if entity in candidates:
            masked_sq_entity_found += 1

    old_masked_sq_entity_found = 0
    for candidates, entity in zip(old_sq_candidates, masked_sq_entities):
        if entity in candidates:
            old_masked_sq_entity_found += 1

    combined_masked_sq_entity_found = 0
    for candidates, entity in zip(sq_union_candidates, masked_sq_entities):
        if entity in candidates:
            combined_masked_sq_entity_found += 1

    rubq_entity_found = 0
    for candidates, entity in zip(rubq_candidates, rubq_entities):
        if entity in candidates:
            rubq_entity_found += 1

    old_rubq_entity_found = 0
    for candidates, entity in zip(old_rubq_candidates, rubq_entities):
        if entity in candidates:
            old_rubq_entity_found += 1

    combined_rubq_entity_found = 0
    for candidates, entity in zip(rubq_union_candidates, rubq_entities):
        if entity in candidates:
            combined_rubq_entity_found += 1

    print("SQ acc: ", sq_entity_found / len(sq_entities))
    print("Masked SQ acc: ", masked_sq_entity_found / len(masked_sq_entities))
    print("Old SQ acc: ", old_masked_sq_entity_found / len(masked_sq_entities))
    print(
        "Combined SQ acc: ", combined_masked_sq_entity_found / len(masked_sq_entities)
    )

    print("RuBQ acc: ", rubq_entity_found / len(rubq_entities))
    print("Old RuBQ acc: ", old_rubq_entity_found / len(rubq_entities))
    print("Сombined RuBQ acc: ", combined_rubq_entity_found / len(rubq_entities))

    #### Top-1 accuracy
    sq_top_1_entity_found = 0
    for sq_result, entity in zip(sq_results, sq_entities):
        top_entity = max(sq_result, key=lambda x: x["score"].item())["id"]
        if top_entity == entity:
            sq_top_1_entity_found += 1

    sq_masked_top_1_entity_found = 0
    for sq_result, entity in zip(np.array(sq_results)[sq_mask], masked_sq_entities):
        top_entity = max(sq_result, key=lambda x: x["score"].item())["id"]
        if top_entity == entity:
            sq_masked_top_1_entity_found += 1

    rubq_top_1_entity_found = 0
    for rubq_result, entity in zip(rubq_results, rubq_entities):
        top_entity = max(rubq_result, key=lambda x: x["score"].item())["id"]
        if top_entity == entity:
            rubq_top_1_entity_found += 1

    print("")
    print("###########  Top 1 ##########")
    print("")
    print("SQ acc: ", sq_top_1_entity_found / len(sq_entities))
    print("Masked SQ acc: ", sq_masked_top_1_entity_found / len(masked_sq_entities))
    print("RuBQ acc: ", rubq_top_1_entity_found / len(rubq_entities))
