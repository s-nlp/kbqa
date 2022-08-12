import torch
from torch import nn
from seq2seq.utils import dbpedia
from transformers import Seq2SeqTrainer

class Seq2SeqWikidataRedirectsTrainer(Seq2SeqTrainer):
    """
    Overwritting the default Seq2SeqTrainer with redirecting feature.
    Now the loss will be the lowest cross entropy loss when
    calculated against all of the redirects sequences
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # squeeze since batch size is 1
        logits = torch.squeeze(logits)
        labels = torch.squeeze(labels)

        # Some simple post-processing and getting redirects for labels
        decoded_labels = self.tokenizer.decode(labels, skip_special_tokens=True)
        decoded_labels = [lab.strip() for lab in decoded_labels]
        decoded_labels = "".join(decoded_labels)
        redirects = dbpedia(decoded_labels)

        # encode the redirects
        encoded_redirects = [labels]
        for red in redirects:
            # pad to 1024
            tokenized = self.tokenizer.encode(red, max_length=1024, padding="max_length")

            res = torch.LongTensor(tokenized).cuda()
            encoded_redirects.append(res)

        # getting the min entropy score
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        curr_min = float("inf")
        for red in encoded_redirects:
            curr_loss = loss_fct(logits, red)

            if curr_loss.item() < curr_min:
                loss = curr_loss
                curr_min = curr_loss.item()

        return (loss, outputs) if return_outputs else loss
