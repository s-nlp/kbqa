from wikidata.wikidata_redirects import WikidataRedirectsCache
from typing import List, Optional, Union
from tqdm.auto import tqdm


def _get_redirects(
    labels: List[str], wikidata_redirects_cache: WikidataRedirectsCache
) -> List[str]:
    all_redirects = []
    for label in set(labels):
        redirects = wikidata_redirects_cache.get_redirects(label)
        if isinstance(redirects, list):
            all_redirects.extend(redirects)

    return all_redirects


def _is_correct_answer_present(
    target_labels: List[str],
    answer_candidates: List[str],
    wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None,
) -> bool:
    if wikidata_redirects_cache is not None:
        target_redirects = _get_redirects(target_labels, wikidata_redirects_cache)
        target_set = set(target_labels + target_redirects)

        answer_redirects = _get_redirects(answer_candidates, wikidata_redirects_cache)
        answer_candidates_set = set(answer_candidates + answer_redirects)
    else:
        target_set = set(target_labels)
        answer_candidates_set = set(answer_candidates)

    return len(answer_candidates_set.intersection(target_set)) > 0


def recall(
    targets: Union[List[str], List[List[str]]],
    answer_candidates: List[List[str]],
    wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None,
) -> float:
    if len(targets) != len(answer_candidates):
        raise ValueError(
            "number of targets and number of answer candidate sets must be equal"
        )
    score = 0
    for idx, target in enumerate(tqdm(targets, desc="recall")):
        if isinstance(target, str):
            target = [target]
        score += int(
            _is_correct_answer_present(
                target, answer_candidates[idx], wikidata_redirects_cache
            )
        )
    return score / len(targets)
