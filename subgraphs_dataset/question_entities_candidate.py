class QuestionEntitiesCandidates:
    """
    class that holds the questions, entities and candidates
    """

    def __init__(self, question: str):
        self.question = question
        self._entities = None
        self._candidates = None
        self._original_entities = None

    @property
    def entities(self):
        return self._entities

    @property
    def candidates(self):
        return self._candidates

    @property
    def original_entities(self):
        return self._original_entities

    @entities.setter
    def entities(self, new_entities):
        self._entities = new_entities

    @candidates.setter
    def candidates(self, new_candidates):
        self._candidates = new_candidates

    def get_entities(self, dirty_entities, lang):
        """
        return the entity texts from generated_entities_batched
        based on the language
        """
        # in case we need the scores in the future
        self._original_entities = dirty_entities

        en_entities = []
        for entity in dirty_entities:
            entity_text = entity["texts"]
            entity_text_en = None

            # getting our entity in the wanted language
            if len(entity_text) > 1:  # received more than one languages
                for text in entity_text:
                    text = text.split(">>")
                    if text[-1].strip() == lang:
                        entity_text_en = text[0].strip()
            else:  # only 1 lang -> english
                text = entity_text[0].split(">>")
                if text[-1].strip() == lang:
                    entity_text_en = text[0].strip()
            if entity_text_en is not None:
                en_entities.append(entity_text_en)
        return en_entities

    def display(self):
        print()
        print("question:", self.question)
        print("entities:", self._entities)
        print("candidates:", self._candidates)
        print()
