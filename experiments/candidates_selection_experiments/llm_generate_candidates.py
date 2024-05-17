import os
import argparse
import torch
import random
import pickle
import datasets
import huggingface_hub

import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
from utils import fix_seed


def main(model_name, file_name, path_checkpoint, num_beams, batch_size, device):
    fix_seed(13)
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        model,
        path_checkpoint,
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset('AmazonScience/mintaka', trust_remote_code=True)

    def generate(data, model=model, tokenizer=tokenizer):
        data = tokenizer(data, return_tensors="pt")['input_ids'].to(device)
        output_ids = model.generate(data, 
            do_sample = False,
            num_beams = num_beams,
            num_beam_groups = num_beams // 10,
            num_return_sequences = num_beams,
            max_new_tokens = 128,
            diversity_penalty = 0.2,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id
        )
        output_ids = [j[data.shape[1]:] for j in output_ids]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output
    
    prompts = []
    prompt = '<s>[INST] Answer as briefly as possible without additional information.\n{message}? [/INST]'
    split = 'train'
    for i in range(len(dataset[split])):
        prompts.append(prompt.format(message=dataset[split][i]['question']))
        
    try:
        with open(file_name, 'rb') as file:
            mistral_answer = pickle.load(file)
    except:
        mistral_answer = []
    for i in tqdm(range(len(mistral_answer), len(dataset[split]))):
        mistral_answer += [{
            'id': dataset[split][i]['id'],
            'question': dataset[split][i]['question'],
            'true_answer': dataset[split][i]['answerText'],
            'model_answer': generate(prompts[i])
        }]
        
        with open(file_name, 'wb') as file:
            pickle.dump(mistral_answer, file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument("--file_name", default='generated_candidates/finetune_mixtral_train.pkl')
    parser.add_argument("--path_checkpoint", default='mixtral_mintaka_checkpoints/final_checkpoint')
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    main(**vars(args))
    