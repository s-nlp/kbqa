import torch
from torch import nn
from transformers import Seq2SeqTrainer
import numpy as np


class Seq2SeqWikidataRedirectsTrainer(Seq2SeqTrainer):
    """
    Overwritting the default Seq2SeqTrainer with redirecting feature.
    Now the loss will be the lowest cross entropy loss when
    calculated against all of the redirects sequences
    """

    def __init__(
        self,
        redirect_cache,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.redirect_cache = redirect_cache

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits = torch.squeeze(logits).cuda()
        labels = torch.squeeze(labels)

        # Some simple post-processing and getting redirects for labels
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [lab.strip() for lab in decoded_labels]

        redirects = [
            self.redirect_cache.get_redirects(decoded_label)
            for decoded_label in decoded_labels
        ]
        max_redirects_length = max(map(len, redirects))

        # encode the redirects
        encoded_redirects_shape = np.zeros(
            (len(redirects), max_redirects_length, logits.size(1))
        )
        encoded_redirects = torch.tensor(encoded_redirects_shape)

        for i, reds in enumerate(redirects):
            if all(isinstance(red, str) for red in reds):
                encoded_reds = []
                for red in reds:
                    tokenized = self.tokenizer.encode(
                        red,
                        padding="max_length",
                        max_length=logits.size(1),
                        truncation=True,
                    )
                    encoded_reds.append(tokenized)

                res = torch.Tensor(torch.Tensor(encoded_reds)).cuda()
                pad = torch.zeros(
                    (max_redirects_length - res.shape[0]), logits.size(1)
                ).cuda()
                res = torch.vstack((res, pad))
                encoded_redirects[i] = res

        # getting the min entropy score
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        curr_min = float("inf")
        encoded_redirects = torch.permute(encoded_redirects, (0, 2, 1)).cuda()
        for i in range(encoded_redirects.size(2)):
            curr_loss = loss_fct(logits[:, :, -1], encoded_redirects[:, :, i])

            if curr_loss.item() < curr_min:
                loss = curr_loss
                curr_min = curr_loss.item()

        return (loss, outputs) if return_outputs else loss
