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
    label_preprocessor_fn: callable = lambda l: l,
) -> bool:
    if wikidata_redirects_cache is not None:
        target_redirects = _get_redirects(target_labels, wikidata_redirects_cache)
        target_labels = target_labels + target_redirects

        answer_redirects = _get_redirects(answer_candidates, wikidata_redirects_cache)
        answer_candidates = answer_candidates + answer_redirects

    target_set = {label_preprocessor_fn(l) for l in target_labels}
    answer_candidates_set = {label_preprocessor_fn(l) for l in answer_candidates}

    return len(answer_candidates_set.intersection(target_set)) > 0


def recall(
    targets: Union[List[str], List[List[str]]],
    answer_candidates: List[List[str]],
    wikidata_redirects_cache: Optional[WikidataRedirectsCache] = None,
    label_preprocessor_fn: callable = lambda l: l,
    verbose: bool = False,
) -> float:
    if len(targets) != len(answer_candidates):
        raise ValueError(
            "number of targets and number of answer candidate sets must be equal"
        )
    if verbose:
        iterate_targets = tqdm(enumerate(targets), total=len(targets), desc="recall")
    else:
        iterate_targets = enumerate(targets)

    score = 0
    for idx, target in iterate_targets:
        if isinstance(target, str):
            target = [target]
        score += int(
            _is_correct_answer_present(
                target,
                answer_candidates[idx],
                wikidata_redirects_cache,
                label_preprocessor_fn,
            )
        )

    return score / len(targets)
