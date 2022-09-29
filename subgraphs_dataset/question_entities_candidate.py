class QuestionEntitiesCandidates:
    """
    class that holds the questions, entities and candidates
    """

    def __init__(self, question: str):
        self.question = question
        self.entity_texts = []
        self.entity_ids = []
        self.candidate_texts = []
        self.candidate_ids = []
        self.original_entities = None

    def get_entities(self, dirty_entities, lang):
        """
        return the entity texts from generated_entities_batched
        based on the language
        """
        # in case we need the scores in the future
        self.original_entities = dirty_entities

        for entity in dirty_entities:
            entity_id = entity["id"]
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
                self.entity_texts.append(entity_text_en)

            self.entity_ids.append(entity_id)

    def populate_candidates(self, candidate_texts, label2entity):
        """
        given the list of natural language candidates, set both texts
        and id's in our class
        """
        self.candidate_texts = candidate_texts

        # get the candidate id
        for candidate in candidate_texts:
            curr_id = label2entity.get_id(candidate)
            self.candidate_ids.append(curr_id)

    def display(self):
        print()
        print("question:", self.question)
        print("entities:", self.entity_ids)
        print("candidates:", self.candidate_ids)
        print()
