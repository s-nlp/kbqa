# pylint: disable=arguments-differ

import logging
import os

import gradio as gr
from transformers import Pipeline, T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Seq2SeqCandidateGeneratorPipeline(Pipeline):
    """Seq2SeqCandidateGeneratorPipeline - HF Pipiline for generatng set of candidates for QA problem
    Working with ConditionalGeneration HF models
    """

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        if "num_beams" in kwargs:
            forward_kwargs["num_beams"] = kwargs.get("num_beams", 200)
        if "num_return_sequences" in kwargs:
            forward_kwargs["num_return_sequences"] = kwargs.get(
                "num_return_sequences", 200
            )
        if "num_beam_groups" in kwargs:
            forward_kwargs["num_beam_groups"] = kwargs.get("num_beam_groups", 20)
        if "diversity_penalty" in kwargs:
            forward_kwargs["diversity_penalty"] = kwargs.get("diversity_penalty", 0.1)
        return {}, forward_kwargs, {}

    def preprocess(self, input_):
        return self.tokenizer(
            input_,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

    def _forward(
        self,
        input_tensors,
        num_beams=200,
        num_return_sequences=200,
        num_beam_groups=20,
        diversity_penalty=0.1,
    ):
        outputs = self.model.generate(
            input_tensors["input_ids"].to(self.device),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )
        return outputs

    def postprocess(self, model_outputs):
        candidates = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return list(dict.fromkeys(candidates))


if __name__ == "__main__":
    with open("examples.txt", "r") as file:
        examples = [e.replace("\n", "") for e in file.readlines()]
    logger.info("Examples loaded: " + "\n".join(examples))

    model_path = os.environ.get(
        "CANDIDATE_GENERATION_MODEL_PATH",
        "/workspace/runs/mintaka_tunned/google_t5-large-ssm-nq/models/checkpoint-3500/",
    )
    tokenizer_path_or_name = os.environ.get(
        "CANDIDATE_GENERATION_TOKENIZER_PATH_OR_NAME", "t5-large"
    )

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path_or_name)

    candidate_generator = Seq2SeqCandidateGeneratorPipeline(
        model=model,
        tokenizer=tokenizer,
    )

    candidate_generation_demo = gr.Interface(
        fn=candidate_generator,
        inputs="text",
        outputs=gr.JSON(),
        title="Candidate Genearion for QA",
        description="T5 Large SSM NQ for Candidate Generation for Question Answerring problem",
        examples=examples,
        cache_examples=True,
        analytics_enabled=True,
    )

    candidate_generation_demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        enable_queue=True,
    )
