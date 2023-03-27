from typing import List, Union
import re
from ..wikidata.wikidata_entity_to_label import WikidataEntityToLabel

import logging


def entities_to_labels(
    entities: Union[List[str], str], entity2label: WikidataEntityToLabel
) -> List[str]:
    """Convert list of objects to list of corresponding labels

    Args:
        entities (Union[List[str], str]): _description_
        entity2label (WikidataEntityToLabel): _description_

    Returns:
        List[str]: list of
    """
    if isinstance(entities, str):
        entities = [entities]

    for idx, entity in enumerate(entities):
        if re.fullmatch(r"Q[0-9]+", entity) is not None:
            entities[idx] = entity2label.get_label(entity)

    return entities


def get_default_logger():
    logger = logging.getLogger("KBQA")
    logger.setLevel(logging.INFO)
    return logger
