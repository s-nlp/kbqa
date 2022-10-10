"""Module for NER insertion to sentence"""

import spacy
from caches.base import CacheBase


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
