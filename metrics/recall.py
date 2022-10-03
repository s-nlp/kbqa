from wikidata.wikidata_redirects import WikidataRedirectsCache
from typing import List, Optional
from tqdm.auto import tqdm


def _is_correct_answer_present(
    target_label: str,
    answer_candidates: List[str],
    wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None,
) -> bool:
    if wikidata_redirects_cache is not None:
        redirects = wikidata_redirects_cache.get_redirects(target_label)
        if not isinstance(redirects, list):
            redirects = []
        target_set = set([target_label] + redirects)

        _answer_candidates_set = set(answer_candidates)
        answer_candidates_set = []
        for answer_candidate in _answer_candidates_set:
            redirects = wikidata_redirects_cache.get_redirects(answer_candidate)
            if isinstance(redirects, list):
                answer_candidates_set.extend(redirects)
        answer_candidates_set = set(answer_candidates_set)
    else:
        target_set = set([target_label])
        answer_candidates_set = set(answer_candidates)

    return len(set(answer_candidates).intersection(target_set)) > 0


def recall(
    targets: List[str],
    answer_candidates: List[List[str]],
    wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None,
) -> float:
    if len(targets) != len(answer_candidates):
        raise ValueError(
            "number of targets and number of answer candidate sets must be equal"
        )

    score = 0
    for idx, target in enumerate(tqdm(targets, desc="recall")):
        score += int(
            _is_correct_answer_present(
                target, answer_candidates[idx], wikidata_redirects_cache
            )
        )
    return score / len(targets)
