import numpy as np
import torch
import datasets
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_name', choices=['bart-base', 't5-small'],
)
parser.add_argument(
    '--dataset', default='../../wikidata_simplequuestion'
)
parser.add_argument(
    '--dataset_config_name', default='answerable_en'
)
parser.add_argument(
    '--dataset_cache_dir', default='../../datasets/'
)
parser.add_argument(
    '--save_dir', default='./runs'
)


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['question'], pad_to_max_length=True, max_length=1024, truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['object'], pad_to_max_length=True, max_length=1024, truncation=True)
    
    labels = target_encodings['input_ids']
    
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels,
    }

    return encodings


def get_model_and_tokenizer_by_name(model_name):
    if model_name == 'bart-base':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    elif model_name == 't5-small':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
    else:
        raise ValueError(f'model_name must be BART or T5, but passed {model_name}')

    return model, tokenizer



def fit_model(args):
    model, tokenizer = get_model_and_tokenizer_by_name(args.model_name)

    dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.dataset_cache_dir,
    )
    dataset = dataset.filter(lambda example: isinstance(example['object'], str))
    dataset = dataset.map(
        lambda batch: convert_to_features(batch, tokenizer),
        batched=True,
    )
    columns = ['input_ids', 'labels', 'attention_mask',] 
    dataset.set_format(type='torch', columns=columns)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.save_dir, f'./models/{args.model_name}'),          
        num_train_epochs=1,           
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,   
        warmup_steps=500,               
        weight_decay=0.01,              
        logging_dir=os.path.join(args.save_dir, f'./logs/{args.model_name}'),
        evaluation_strategy='steps',
        eval_steps=250,
        logging_steps=250,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,                       
        args=training_args,                  
        train_dataset=dataset['train'],        
        eval_dataset=dataset['validation']   
    )

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    fit_model(args)
