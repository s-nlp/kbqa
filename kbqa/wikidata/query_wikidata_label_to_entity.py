""" query wikidata services to get label to entity"""
from pywikidata.utils import get_wd_search_results


def label_to_entity(label: str, top_k: int = 1) -> list:
    """label_to_entity method to  linking label to WikiData entity ID
    by using elasticsearch Wikimedia public API
    Supported only English language (en)

    Parameters
    ----------
    label : str
        label of entity to search
    top_k : int, optional
        top K results from WikiData, by default 1

    Returns
    -------
    list[str] | None
        list of entity IDs or None if not found
    """
    try:
        elastic_results = get_wd_search_results(label, top_k, language="en")[:top_k]
    except:  # pylint: disable=bare-except
        elastic_results = []

    try:
        elastic_results.extend(
            get_wd_search_results(
                label.replace('"', "").replace("'", "").strip(), top_k, language="en"
            )[:top_k]
        )
    except:  # pylint: disable=bare-except
        return [None]

    if len(elastic_results) == 0:
        return [None]

    return list(dict.fromkeys(elastic_results).keys())[:top_k]
