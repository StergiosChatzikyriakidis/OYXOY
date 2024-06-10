from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import transformers
import torch
import pandas as pd
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.llms.oyxoy_datasets import PromptHandlerSelector
import argparse
from pathlib import Path




def main(args):
    device = "cpu" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.to(device)

    
    prompt_handler = PromptHandlerSelector(args.dataset).create(args.prompt, args.num_shots)

    for test_set_name, test_set in prompt_handler.test_sets.items():
        Path(args.output_dir_path+'/'+test_set_name).mkdir(parents=True, exist_ok=True)
        for ii, sample in tqdm(enumerate(test_set)):

            messages = prompt_handler.get_messages(sample, ii)

            
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_prompt = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model.generate(
                input_prompt['input_ids'], 
                max_new_tokens=args.max_new_tokens, 
                do_sample=True, 
                temperature=args.temperature
            )

            series = pd.DataFrame(test_set).iloc[ii]
            series['output'] = tokenizer.batch_decode(outputs)[0][6+len(prompt):-4]
            series.to_csv(f'{args.output_dir_path}/{test_set_name}/llama3_{ii}.csv')
            print(tokenizer.batch_decode(outputs)[0][6+len(prompt):-4])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="ilsp/Meltemi-7B-Instruct-v1", help="pretrained model from HF that can be used for llama prompting")
    parser.add_argument('-out', '--output_dir_path', required=True, help='Directory path where output .csv files will be stored') 
    parser.add_argument('-prompt', '--prompt', required=True, help='Prompt method from the available in `experiments/llms/prompts.py`')
    parser.add_argument('-temp', '--temperature', required=False, default=0.6, help='temperature argument for the LLM')
    parser.add_argument('-n_tokens', '--max_new_tokens', type=int, required=False, default=4, help='Maximum number of new tokens (following prompt) to generate')
    parser.add_argument('-n_shots', '--num_shots', type=int, required=False, default=None, help='Number of examples to use for the ICL')
    parser.add_argument('-ds', '--dataset', type=str, required=False, default=None, help='Dataset on which to run the experiment')

    args = parser.parse_args()
  
    assert (('few_shot' not in args.prompt) and (args.num_shots is None)) or (('few_shot' in args.prompt) and (args.num_shots is not None)), "You selected a few-shot prompt so you should provide an integer value for `-n_shots` "
    main(args)
