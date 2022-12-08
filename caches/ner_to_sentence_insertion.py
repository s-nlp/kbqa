"""Module for NER insertion to the sentence"""

import spacy

from caches.base import CacheBase
from config import DEFAULT_CACHE_PATH


class NerToSentenceInsertion(CacheBase):
    """Module for adding START and END tokens"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        model_path: str = ".spacy-finetuned/output/model-best",
    ) -> None:

        super().__init__(cache_dir_path, "wikidata_with_ner.pkl")
        self.cache = {}
        self.load_from_cache()

        self.model = spacy.load(model_path)

    def entity_labeling(self, test_question, get_num_entities=False):
        """First lettters capitalization and START/END tokens for entities insertion"""

        if test_question not in self.cache:
            # ner part
            nlp = self.model
            doc = nlp(test_question)
            entities_list = [ent.text for ent in doc.ents]
            num_entities = len(entities_list)
            if num_entities > 0:
                ner_question = test_question
                for entity in entities_list:
                    entity_index_in_string = ner_question.find(entity)
                    ner_question = (
                        ner_question[:entity_index_in_string]
                        + " [START] "
                        + ner_question[
                            entity_index_in_string : entity_index_in_string
                            + len(entity)
                        ]
                        + " [END] "
                        + ner_question[entity_index_in_string + len(entity) :]
                    ).replace("  ", " ")
            else:
                ner_question = "[START] " + str(test_question) + " [END]"

            # LargeCase part
            sent_split = []
            for elem in ner_question.split(" "):
                if elem != "":
                    sent_split.append(elem[0].upper() + elem[1:])
            ner_largecase_question = " ".join(sent_split)

            self.cache[test_question] = (ner_largecase_question, num_entities)
            self.save_cache()
        else:
            ner_largecase_question, num_entities = self.cache[test_question]

        if get_num_entities:
            return ner_largecase_question, num_entities
        return ner_largecase_question
