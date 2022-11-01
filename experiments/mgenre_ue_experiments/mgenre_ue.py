"""UE calculations for single model and ensemble of mGENRE models"""

import pickle
import torch
import string
import math
from itertools import permutations
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from scipy.special import kl_div
import pandas as pd

from mgenre_eop_eval import get_top_ensemble_prediction, get_all_ensemble_predictions

MASTER_SEED = 42


def get_top_result_with_score(results):
    """Extracts top prediction along with it's score from mGENRE results"""
    top_res = max(results, key=lambda x: x["score"].item())
    return top_res["id"], top_res["score"].item()


def get_second_result_with_score(results):
    """Extracts top-2 prediction along with it's score from mGENRE results"""
    if len(results) > 1:
        second_res = sorted(results, key=lambda x: x["score"].item())[-2]
        return second_res["id"], second_res["score"].item()
    
    return None, None


def get_delta_ue(results):
    """Calculates difference between top-1 and top-2 scores"""
    top_score = get_top_result_with_score(results)[1]
    second_score = get_second_result_with_score(results)[1]

    if second_score is not None:
        return top_score - second_score

    return top_score


def get_top_1_acc(mgenre_results, gt_entities):
    """Calculates top-1 accuracy of provided mGENRE results"""
    top_1_entity_found = 0
    for result, entity in zip(mgenre_results, gt_entities):
        top_entity = max(result, key=lambda x: x["score"].item())["id"]
        if type(entity) == list:
            if top_entity in entity:
                top_1_entity_found += 1
        else:
            if top_entity == entity:
                top_1_entity_found += 1

    return top_1_entity_found / len(gt_entities)


def get_top_1_ensemble_acc(ensemble_results, gt_entities):
    """Calculates top-1 accuracy of provided mGENRE ensemble results"""
    top_1_entity_found = 0
    for result, entity in zip(ensemble_results, gt_entities):
        top_entity = get_top_ensemble_prediction(result)
        if type(entity) == list:
            if top_entity in entity:
                top_1_entity_found += 1
        else:
            if top_entity == entity:
                top_1_entity_found += 1

    return top_1_entity_found / len(gt_entities)


def reject_data(ue, preds, gts, rate, flip=True):
    """Performs rejection procedure based on provided ue for provided predictions"""
    data_len = len(gts)
    num_points_after_rejection = data_len - int(rate * data_len)
    if flip:
        order = np.flip(np.argsort(ue))
    else:
        order = np.argsort(ue)
    ordered_preds = np.array(preds)[order]
    ordered_gts = np.array(gts)[order]

    retained_preds = ordered_preds[0:num_points_after_rejection]
    retained_gts = ordered_gts[0:num_points_after_rejection]

    return retained_preds, retained_gts


def get_rubq_question_entities(uris):
    """Extracts gt entities from RuBQ WD uris"""
    ents = []
    if type(uris) == list:
        for uri in uris:
            ents.append(uri.split("/")[-1])
    else:
        ents = [None]

    return ents


def mgenre_single_ue(sq_results_file, rubq_results_file, rubq_full=False):
    """Calculates score-based and delta-based UE and performs rejection"""
    if rubq_full:
        rubq_entities = pd.read_json("RuBQ_2.0_test.json")["question_uris"].apply(
            get_rubq_question_entities
        )
    else:
        rubq_entities = np.load("rubq_test_entities.npy")

    sq_test = np.load("./data/simple_questions_test.npy")
    sq_mask = np.load("./data/sq_mask_present_in_graph.npy")

    sq_entities = []
    sq_questions = []
    for (entity, _, _, question) in sq_test:
        sq_entities.append(entity)
        sq_questions.append(question)
    masked_sq_entities = np.array(sq_entities)[sq_mask]

    sq_results = None
    rubq_results = None

    with open(sq_results_file, "rb") as handle:
        sq_results = pickle.load(handle)

    with open(rubq_results_file, "rb") as handle:
        rubq_results = pickle.load(handle)

    masked_sq_results = np.array(sq_results)[sq_mask]

    #### Top-1 accuracy
    print("")
    print("###########  Top 1 ##########")
    print("")
    print("SQ acc: ", get_top_1_acc(sq_results, sq_entities))
    print("Masked SQ acc: ", get_top_1_acc(masked_sq_results, masked_sq_entities))
    print("RuBQ acc: ", get_top_1_acc(rubq_results, rubq_entities))

    sq_scores = [get_top_result_with_score(result)[1] for result in sq_results]
    sq_masked_scores = [
        get_top_result_with_score(result)[1] for result in masked_sq_results
    ]
    rubq_scores = [get_top_result_with_score(result)[1] for result in rubq_results]

    sq_deltas = [get_delta_ue(result) for result in sq_results]
    sq_masked_deltas = [get_delta_ue(result) for result in masked_sq_results]
    rubq_deltas = [get_delta_ue(result) for result in rubq_results]

    rejection_rates = np.linspace(0, 0.9, 20)

    sq_score_accs = []
    masked_sq_score_accs = []
    rubq_score_accs = []

    sq_delta_accs = []
    masked_sq_delta_accs = []
    rubq_delta_accs = []

    for rate in rejection_rates:
        ### SQ scores
        retained_results, retained_entities = reject_data(
            sq_scores, sq_results, sq_entities, rate
        )
        sq_score_accs.append(get_top_1_acc(retained_results, retained_entities))

        ### Masked SQ scores
        retained_results, retained_entities = reject_data(
            sq_masked_scores, masked_sq_results, masked_sq_entities, rate
        )
        masked_sq_score_accs.append(get_top_1_acc(retained_results, retained_entities))

        ### RuBQ scoresd
        retained_results, retained_entities = reject_data(
            rubq_scores, rubq_results, rubq_entities, rate
        )
        rubq_score_accs.append(get_top_1_acc(retained_results, retained_entities))

        ### SQ deltas
        retained_results, retained_entities = reject_data(
            sq_deltas, sq_results, sq_entities, rate
        )
        sq_delta_accs.append(get_top_1_acc(retained_results, retained_entities))

        ### Masked SQ deltas
        retained_results, retained_entities = reject_data(
            sq_masked_deltas, masked_sq_results, masked_sq_entities, rate
        )
        masked_sq_delta_accs.append(get_top_1_acc(retained_results, retained_entities))

        ### RuBQ deltas
        retained_results, retained_entities = reject_data(
            rubq_deltas, rubq_results, rubq_entities, rate
        )
        rubq_delta_accs.append(get_top_1_acc(retained_results, retained_entities))

    return (
        rejection_rates,
        sq_score_accs,
        masked_sq_score_accs,
        rubq_score_accs,
        sq_delta_accs,
        masked_sq_delta_accs,
        rubq_delta_accs,
    )


def get_ue(total_preds):
    """Calculates entropy-based UE"""
    eentropies, pentropies, balds, epkls, rmis = [], [], [], [], []
    eps = 1e-10

    for res in total_preds:
        all_ids = []

        for model_res in res:
            for pred in model_res:
                all_ids.append(pred["id"])
        all_ids = set(all_ids)

        if len(all_ids) == 0:
            return (0.0, 0.0, 0.0)

        scores = []
        ids = []

        for model_res in res:
            model_ids = []
            for pred in model_res:
                model_ids.append(pred["id"])
            model_ids = set(model_ids)

            missing_ids = all_ids - model_ids

            for _id in missing_ids:
                model_res.append({"id": _id, "score": -math.inf})

            sorted_model_res = sorted(model_res, key=lambda x: x["id"])

            model_scores = [pred["score"] for pred in sorted_model_res]
            model_ids = [pred["id"] for pred in sorted_model_res]
            scores.append(np.array(model_scores))
            ids.append(np.array(model_ids))

        probas = np.exp(np.stack(scores))
        ids = np.stack(ids)

        permutations_of_models = permutations(range(len(probas)), 2)
        kls = []
        for i, j in permutations_of_models:
            kls.append(sum(kl_div(probas[i] + eps, probas[j] + eps)))
        epkl = np.mean(kls, axis=0)

        average_probas = probas.mean(axis=0)

        pentropy = entropy(average_probas)
        eentropy = np.mean([entropy(model_probas) for model_probas in probas])
        bald = pentropy - eentropy
        rmi = epkl - bald

        pentropies.append(pentropy)
        eentropies.append(eentropy)
        balds.append(bald)
        epkls.append(epkl)
        rmis.append(rmi)

    return (pentropies, eentropies, balds, epkls, rmis)


def mgenre_ensemble_ue(sq_results_file, rubq_results_file, rubq_full=False):
    """Calculates entropy-based UE and performs rejection for ensemble of mGENRE models"""
    if rubq_full:
        rubq_entities = pd.read_json("RuBQ_2.0_test.json")["question_uris"].apply(
            get_rubq_question_entities
        )
    else:
        rubq_entities = np.load("rubq_test_entities.npy")

    sq_test = np.load("./data/simple_questions_test.npy")
    sq_mask = np.load("./data/sq_mask_present_in_graph.npy")

    sq_entities = []
    sq_questions = []
    for (entity, _, _, question) in sq_test:
        sq_entities.append(entity)
        sq_questions.append(question)
    masked_sq_entities = np.array(sq_entities)[sq_mask]

    sq_results = None
    rubq_results = None

    with open(sq_results_file, "rb") as handle:
        sq_results = pickle.load(handle)

    with open(rubq_results_file, "rb") as handle:
        rubq_results = pickle.load(handle)

    masked_sq_results = np.array(sq_results)[sq_mask]

    #### Top-1 accuracy
    print("")
    print("###########  Top 1 ##########")
    print("")
    print("SQ acc: ", get_top_1_ensemble_acc(sq_results, sq_entities))
    print(
        "Masked SQ acc: ", get_top_1_ensemble_acc(masked_sq_results, masked_sq_entities)
    )
    print("RuBQ acc: ", get_top_1_ensemble_acc(rubq_results, rubq_entities))

    sq_eentropies, sq_pentropies, sq_balds, sq_epkls, sq_rmis = get_ue(sq_results)
    rubq_eentropies, rubq_pentropies, rubq_balds, rubq_epkls, rubq_rmis = get_ue(
        rubq_results
    )

    rejection_rates = np.linspace(0, 0.9, 20)

    rejected_sq_accs = [[], [], [], [], []]
    rejected_masked_accs = [[], [], [], [], []]
    rejected_rubq_accs = [[], [], [], [], []]

    for rate in rejection_rates:
        ### SQ
        for i, ue in enumerate(
            [sq_eentropies, sq_pentropies, sq_balds, sq_epkls, sq_rmis]
        ):
            retained_results, retained_entities = reject_data(
                ue, sq_results, sq_entities, rate, flip=False
            )
            rejected_sq_accs[i].append(
                get_top_1_ensemble_acc(retained_results, retained_entities)
            )

        ### Masked SQ
        for i, ue in enumerate(
            [
                np.array(sq_eentropies)[sq_mask],
                np.array(sq_pentropies)[sq_mask],
                np.array(sq_balds)[sq_mask],
                np.array(sq_epkls)[sq_mask],
                np.array(sq_rmis)[sq_mask],
            ]
        ):
            retained_results, retained_entities = reject_data(
                ue,
                np.array(sq_results)[sq_mask],
                np.array(sq_entities)[sq_mask],
                rate,
                flip=False,
            )
            rejected_masked_accs[i].append(
                get_top_1_ensemble_acc(retained_results, retained_entities)
            )

        ### RuBQ
        for i, ue in enumerate(
            [rubq_eentropies, rubq_pentropies, rubq_balds, rubq_epkls, rubq_rmis]
        ):
            retained_results, retained_entities = reject_data(
                ue, rubq_results, rubq_entities, rate, flip=False
            )
            rejected_rubq_accs[i].append(
                get_top_1_ensemble_acc(retained_results, retained_entities)
            )

    (
        sq_eentropy_accs,
        sq_pentropy_accs,
        sq_bald_accs,
        sq_epkls_acss,
        sq_rmis_accs,
    ) = rejected_sq_accs
    (
        masked_sq_eentropy_accs,
        masked_sq_pentropy_accs,
        masked_sq_bald_accs,
        masked_sq_epkls_acss,
        masked_sq_rmis_accs,
    ) = rejected_masked_accs
    (
        rubq_eentropy_accs,
        rubq_pentropy_accs,
        rubq_bald_accs,
        rubq_epkls_acss,
        rubq_rmis_accs,
    ) = rejected_rubq_accs

    return (
        rejection_rates,
        sq_eentropy_accs,
        sq_pentropy_accs,
        sq_bald_accs,
        sq_epkls_acss,
        sq_rmis_accs,
        masked_sq_eentropy_accs,
        masked_sq_pentropy_accs,
        masked_sq_bald_accs,
        masked_sq_epkls_acss,
        masked_sq_rmis_accs,
        rubq_eentropy_accs,
        rubq_pentropy_accs,
        rubq_bald_accs,
        rubq_epkls_acss,
        rubq_rmis_accs,
    )
