from transformers import Pipeline


class MGENREPipeline(Pipeline):
    """MGENREPipeline - HF Pipeline for mGENRE EntityLinking model"""

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        if "num_beams" in kwargs:
            forward_kwargs["num_beams"] = kwargs.get("num_beams", 10)
        if "num_return_sequences" in kwargs:
            forward_kwargs["num_return_sequences"] = kwargs.get(
                "num_return_sequences", 10
            )
        return {}, forward_kwargs, {}

    def preprocess(self, input_):
        return self.tokenizer(
            input_,
            return_tensors="pt",
        )

    def _forward(
        self,
        input_tensors,
        num_beams=10,
        num_return_sequences=10,
    ):
        outputs = self.model.generate(
            **{k: v.to(self.device) for k, v in input_tensors.items()},
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        return outputs

    def postprocess(self, model_outputs):
        outputs = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs
