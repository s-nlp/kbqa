"""Module for detection True case of words in sentence"""


import warnings
import os
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from pytorch_truecaser.mylib import truecaser_predictor
from base import CacheBase

warnings.filterwarnings("ignore")
os.rename("pytorch-truecaser", "pytorch_truecaser")


class TrueCaseDetection(CacheBase):
    """Module for detection True case of words in sentence"""

    def __init__(
        self,
        cache_dir_path: str = "./cache_store",
    ) -> None:

        super().__init__(cache_dir_path, "truecase.pkl")
        self.cache = {}
        self.load_from_cache()

    def truecase_withoutner_mayhewsw(self, sentence):
        """True case detection"""

        if sentence not in self.cache:
            archive = load_archive("wiki-truecaser-model-en.tar.gz")
            predictor = Predictor.from_archive(archive, "truecaser-predictor")
            out = predictor.predict(sentence)
            outline = predictor.dump_line(out).split("\n")[0]
            sentence_withoutner = "[START] " + str(outline) + " [END]"
            self.cache[sentence] = sentence_withoutner
            self.save_cache()
        else:
            sentence_withoutner = self.cache[sentence]

        return sentence_withoutner

    def truecase_withner_mayhewsw(self, sentence_withner, sentence_truecase):

        """True case detection with NER"""
        sent_to_change = sentence_withner.replace("  ", " ")
        sent_to_change = sent_to_change.split(" ")
        if "" in sent_to_change:
            sent_to_change.remove("")
        sent_truecase = sentence_truecase.replace("  ", " ")
        sent_truecase = sent_truecase[8:-6].split(" ")
        if "" in sent_truecase:
            sent_truecase.remove("")

        if (
            "[START]  [END]" not in sent_to_change
            and len(sent_to_change) == len(sent_truecase) + 2
        ):
            for ind, letter_pos in enumerate(sent_to_change):
                if letter_pos == "[START]":
                    id_start = ind
                elif letter_pos == "[END]":
                    id_end = ind
            for begin in range(id_start):
                sent_to_change[begin] = sent_truecase[begin]
            for mid in range(id_start + 1, id_end):
                sent_to_change[mid] = sent_truecase[mid - 1]
            for end in range(id_end + 1, len(sent_to_change) - 1):
                sent_to_change[end] = sent_truecase[end - 2]
            return " ".join(sent_to_change)
        return " ".join(sent_truecase)

    def predictions_reranking(self, pred_main, pred_secondary):
        """Module for the predictions reranking"""

        # pred_main, pred_secondary - string of predictions separated by comma

        preds_main = pred_main.split(", ")
        preds_secondary = pred_secondary.split(", ")
        preds_joined = []
        min_index = min(len(preds_main), len(preds_secondary))

        # if the length of pred_main and pred_secondary coincide
        for k in range(min_index):
            if preds_main[k] == preds_secondary[k]:
                preds_joined.append(preds_main[k])
            else:
                preds_joined.append(preds_main[k])
                preds_joined.append(preds_secondary[k])

        # if pred_main larger than pred_secondary
        if len(preds_main) > len(preds_secondary):
            diff = len(preds_main) - len(preds_secondary)
            subset = preds_main[-diff:]
            for _, elem in enumerate(subset):
                preds_joined.append(elem)

        # if pred_secondary larger than pred_main
        elif len(preds_main) < len(preds_secondary):
            diff = len(preds_secondary) - len(preds_main)
            subset = preds_secondary[-diff:]
            for _, elem in enumerate(subset):
                preds_joined.append(elem)
        return ", ".join(preds_joined)
