"""Module for NER insertion to the sentence"""

import spacy
import os
import pickle
from abc import ABC


class CacheBase(ABC):
    """CacheBase - Abstract base class for storing something in cache file"""

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
        cache_filename: str = "cache.pkl",
    ) -> None:
        self.cache_dir_path = cache_dir_path
        self.cache_filename = cache_filename
        self.cache = None

        self.cache_file_path = os.path.join(self.cache_dir_path, self.cache_filename)

        self.load_from_cache()

    def load_from_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "rb") as file:
                self.cache = pickle.load(file)

    def save_cache(self):
        if not os.path.exists(self.cache_dir_path):
            os.makedirs(self.cache_dir_path)

        with open(self.cache_file_path, "wb") as file:
            pickle.dump(self.cache, file)


class NerToSentenceInsertion(CacheBase):
    """Module for adding START and END tokens"""

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
        model_path: str = ".spacy-finetuned/output/model-best",
    ) -> None:

        super().__init__(cache_dir_path, "wikidata_with_ner.pkl")
        self.cache = {}
        self.load_from_cache()

        self.model = spacy.load(model_path)

    def entity_labeling(self, test_question):
        """First lettters capitalization and START/END tokens for entities insertion"""

        if test_question not in self.cache:

            # NER part

            nlp = self.model
            doc = nlp(test_question)
            entities = ",".join([ent.text for ent in doc.ents])
            if entities != "":
                index = test_question.find(entities)
                ner_question = (
                    test_question[:index]
                    + "[START] "
                    + test_question[index : index + len(entities)]
                    + " [END]"
                    + test_question[index + len(entities) :]
                )
            else:
                ner_question = "[START] " + str(test_question) + " [END]"

            # LargeCase part

            sent_split = []
            for elem in ner_question.split(" "):
                if elem != "":
                    sent_split.append(elem[0].upper() + elem[1:])
            ner_largecase_question = " ".join(sent_split)

            self.cache[test_question] = ner_largecase_question
            self.save_cache()
        else:
            ner_largecase_question = self.cache[test_question]

        return ner_largecase_question
