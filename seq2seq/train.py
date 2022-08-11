from typing import Tuple, Optional
import datasets
from seq2seq.utils import load_model_and_tokenizer_by_name, load_kbqa_seq2seq_dataset
import torch.nn as nn
import torch
import os
import sys
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF

from transformers import (
    Trainer,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# return the array of redirects
def dbpedia(term):
    term = term.strip()
    nterm = term.capitalize().replace(" ", "_")

    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label
    WHERE 
    { 
     {
     <http://dbpedia.org/resource/VALUE> <http://dbpedia.org/ontology/wikiPageRedirects> ?x.
     ?x rdfs:label ?label.
     }
     UNION
     { 
     <http://dbpedia.org/resource/VALUE> <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
     ?x <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
     ?x rdfs:label ?label.
     }
    UNION
     {
     ?x <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/VALUE>.
     ?x rdfs:label ?label.
     }
     UNION
     { 
     ?y <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/VALUE>.
     ?x <http://dbpedia.org/ontology/wikiPageRedirects> ?y.
     ?x rdfs:label ?label.
     }
     FILTER (lang(?label) = 'en')
    }
    """

    nquery = query.replace("VALUE", nterm)

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(nquery)
    rterms = []
    sparql.setReturnFormat(JSON)
    try:
        ret = sparql.query()
        results = ret.convert()
        requestGood = True
    except Exception:
        results = ""
        requestGood = False

    if requestGood == False:
        return "Problem communicating with the server: ", results
    elif len(results["results"]["bindings"]) == 0:
        return "No results found"
    else:
        for result in results["results"]["bindings"]:
            label = result["label"]["value"]
            rterms.append(label)
    alts = ", ".join(rterms)
    # alts = alts.encode('utf-8')

    # converting back to an array of strings
    res = alts.split(",")
    return res


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # squeeze since batch size is 1
        logits = torch.squeeze(logits)
        labels = torch.squeeze(labels)

        # Some simple post-processing
        decoded_labels = self.tokenizer.decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_labels]
        decoded_preds = "".join(decoded_preds)

        redirects = dbpedia(decoded_preds)
        print(redirects)

        # encode the redirects
        encoded_redirects = [labels]
        for r in redirects:
            # pad to 1024
            tokenized = self.tokenizer.encode(r, max_length=1024, padding="max_length")

            res = torch.LongTensor(tokenized).cuda()
            encoded_redirects.append(res)

        # getting the min entropy score
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        curr_min = float("inf")
        for r in encoded_redirects:
            curr_loss = loss_fct(logits, r)

            if curr_loss.item() < curr_min:
                loss = curr_loss
                curr_min = curr_loss.item()

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decoding the predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    loss = nn.CrossEntropyLoss()
    cross_entropy = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # getting the redirect for the current label
        redirects = dbpedia(label)
        for r in redirects:
            cross_entropy.append(loss(pred, loss))

    return {"lowest_cross_entropy": min(cross_entropy)}


def train(
    model_name: str,
    dataset_name: str,
    dataset_config_name: str,
    dataset_cache_dir: Optional[str] = None,
    output_dir: str = "./runs/models/model",
    logging_dir: str = "./runs/logs/model",
    save_total_limit: int = 5,
    num_train_epochs: int = 8,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    logging_steps: int = 500,
    gradient_accumulation_steps: int = 8,
) -> Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]:
    """train seq2seq model for KBQA problem
    Work with HF dataset with object and question field (str)

    Args:
        model_name (str): HF seq2seq model: SEQ2SEQ_AVAILABLE_HF_PRETRAINED_MODEL_NAMES
        dataset_name (str): name or path to HF dataset with str fields: object, question
        dataset_config_name (str): HF dataset config name
        dataset_cache_dir (str, optional): Path to HF cache. Defaults to None.
        output_dir (str): Path to directory for storing model's checkpoints . Defaults to './runs/models/model'
        logging_dir (str): Path to directory for storing traning logs . Defaults to './runs/logs/model'
        save_total_limit (int, optional): Total limit for storing model's checkpoints. Defaults to 5.
        num_train_epochs (int, optional): Total number of traning epoches. Defaults to 8.
        per_device_train_batch_size (int, optional): train batch size per device. Defaults to 1.
        per_device_eval_batch_size (int, optional): eval batch size per device. Defaults to 1.
        warmup_steps (int, optional): warmup steps for traning. Defaults to 500.
        weight_decay (float, optional): weight decay for traning. Defaults to 0.01.
        evaluation_strategy (str, optional):
            "no": No evaluation is done during training;
            "steps": Evaluation is done (and logged) every eval_steps;
            "epoch": Evaluation is done at the end of each epoch;
            Defaults to 'steps'.
        eval_steps (int, optional):
            Number of update steps between two evaluations if evaluation_strategy="steps".
            Will default to the same value as logging_steps if not set.
            Defaults to 500.
        logging_steps (int, optional):
            Number of update steps between two logs if logging_strategy="steps".
            Defaults to 500.
        gradient_accumulation_steps (int, optional):
             Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
             Defaults to 8.

    Returns:
        Tuple[Seq2SeqTrainer, PreTrainedModel, datasets.arrow_dataset.Dataset]: _description_
    """
    model, tokenizer = load_model_and_tokenizer_by_name(model_name)

    dataset = load_kbqa_seq2seq_dataset(
        dataset_name, dataset_config_name, tokenizer, dataset_cache_dir
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()

    return trainer, model, dataset
