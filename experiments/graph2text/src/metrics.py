from typing import Callable, Dict, List, Union

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from transformers import EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast


def calculate_text_metrics(
    answers: List[str],
    target: List[str],
) -> Dict[str, float]:
    chrf = CHRF()
    chrf_plus_plus = CHRF(word_order=2)
    bleu = BLEU()

    # Replace "" to " " if required.
    # Withot this one hack, sacrebleu raise unexpected exception
    target = [t if len(t) > 0 else " " for t in target]

    target = [target]

    chrf_score = chrf.corpus_score(
        answers,
        target,
    ).score

    chrf_plus_plus_score = chrf_plus_plus.corpus_score(
        answers,
        target,
    ).score

    bleu_score = bleu.corpus_score(
        answers,
        target,
    ).score

    return {
        "chrf": chrf_score,
        "chrf++": chrf_plus_plus_score,
        "bleu": bleu_score,
    }


def build_compute_metrics(
    tokenizer: PreTrainedTokenizer, ignore_pad_token
) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics(eval_prediction: EvalPrediction) -> Dict:
        preds = eval_prediction.predictions
        predictions = np.where(preds != ignore_pad_token, preds, tokenizer.pad_token_id)

        label_ids = eval_prediction.label_ids
        label_ids = np.where(
            label_ids != ignore_pad_token, label_ids, tokenizer.pad_token_id
        )

        generated_text = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        target = tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        return calculate_text_metrics(
            generated_text,
            target,
        )

    return compute_metrics


def chrf_compute_objective(metrics: Dict[str, float]) -> float:
    return metrics["eval_chrf"]
