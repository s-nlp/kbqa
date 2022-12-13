import gradio as gr
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.ner import NerToSentenceInsertion
from src.mgenre import MGENREPipeline
from functools import lru_cache
from nltk.stem.porter import PorterStemmer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EntitiesSelection:
    def __init__(self, ner_model):
        self.stemmer = PorterStemmer()
        self.ner_model = ner_model

    def entities_selection(self, entities_list, mgenre_predicted_entities_list):
        final_preds = []

        for pred_text in mgenre_predicted_entities_list:
            labels = []
            _label, lang = pred_text.split(" >> ")
            if lang == "en":
                labels.append(_label)

            if len(labels) > 0:
                for label in labels:
                    label = label.lower()
                    if self._check_label_fn(label, entities_list):
                        final_preds.append(pred_text)

        return final_preds

    @lru_cache(maxsize=8192)
    def _label_format_fn(self, label):
        return " ".join(
            [self.stemmer.stem(str(token)) for token in self.ner_model(label)]
        )

    def _check_label_fn(self, label, entities_list):
        label = self._label_format_fn(label)
        for entity in entities_list:
            entity = self._label_format_fn(entity)
            if label == entity:
                return True
        return False


class EntityLinker:
    def __init__(
        self,
        ner_model_path: str,
        ner_examples_path: str,
        mgenre_examples_path: str,
        mgenre_num_beams: int,
        mgenre_num_return_sequences: int,
    ):
        self._init_ner(ner_model_path, ner_examples_path)
        self._init_mgenre(mgenre_examples_path, mgenre_num_beams, mgenre_num_return_sequences)
        self.entity_selection_model = EntitiesSelection(self.ner_model)

    def get_ner_interface(
        self,
    ):
        """Return gradio interface for NER model"""

        def __get_ner_results(text):
            text_with_labeling, entities_list = self.ner.entity_labeling(text, True)
            return {
                "text_with_labeling": text_with_labeling,
                "entities": entities_list,
            }

        return gr.Interface(
            fn=__get_ner_results,
            inputs="text",
            outputs=gr.JSON(),
            title="Named Entity Recognition",
            description="Spacy fine tuned model for NER",
            examples=self.ner_examples,
            cache_examples=True,
            analytics_enabled=True,
        )

    def get_mgenre_interface(
        self,
    ):
        """Return gradio interface for mGENRE model"""
        return gr.Interface(
            fn=self.mgenre,
            inputs="text",
            outputs=gr.JSON(),
            title="Entity Linking by mGENRE",
            description="mGENRE for Entity Linking. Use Named Entity Recognition interface for preparing input string for it.",
            examples=self.mgenre_examples,
            cache_examples=True,
            analytics_enabled=True,
        )

    def get_enities_linking_interface(self):
        """Return gradio interface for entity linking
        NER + mGENRE + entities_selection
        """
        return gr.Interface(
            fn=self._entitest_linking,
            inputs="text",
            outputs=gr.JSON(),
            title="Entity Linking by mGENRE with NER and EntitiesSelection",
            description="NER for preparing text for mGENRE; mGENRE for Entity Linking; Base Entities Selection",
            analytics_enabled=True,
        )

    def _entitest_linking(self, text):
        text_with_labeling, entities_list = self.ner.entity_labeling(text, True)
        mgenre_predicted_entities_list = self.mgenre(text_with_labeling)
        linked_entities_list = self.entity_selection_model.entities_selection(
            entities_list, mgenre_predicted_entities_list
        )
        return {
            "text_with_labeling": text_with_labeling,
            "ner_entities": entities_list,
            "mgenre_predicted_entities_list": mgenre_predicted_entities_list,
            "final_linked_entities_list": linked_entities_list,
        }

    def _init_ner(
        self,
        ner_model_path,
        ner_examples_path,
    ):
        with open(ner_examples_path, "r") as file:
            self.ner_examples = [e.replace("\n", "") for e in file.readlines()]
        logger.info("NER examples loaded: " + "\n".join(self.ner_examples))

        self.ner = NerToSentenceInsertion(ner_model_path)
        self.ner_model = self.ner.model

    def _init_mgenre(self, mgenre_examples_path, mgenre_num_beams, mgenre_num_return_sequences):
        with open(mgenre_examples_path, "r") as file:
            self.mgenre_examples = [e.replace("\n", "") for e in file.readlines()]
        logger.info("mGENRE examples loaded: " + "\n".join(self.mgenre_examples))

        tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()
        self.mgenre = MGENREPipeline(
            model=model,
            tokenizer=tokenizer,
            num_beams=mgenre_num_beams,
            num_return_sequences=mgenre_num_return_sequences,
        )
