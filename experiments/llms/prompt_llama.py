from tqdm import tqdm
import transformers
import torch
import pandas as pd
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.inference.data import load_data
import argparse
from experiments.llms.prompts import select_prompt
from pathlib import Path


def main(args):
    model_id = args.model

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=args.hf_token
    )


    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    prompt_sys, prompt_usr = select_prompt(args.prompt)
    _, test = load_data()

    Path(args.output_dir_path).mkdir(parents=True, exist_ok=True)

    for ii, sample in tqdm(enumerate(test)):
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_usr.format(sample.premise, sample.hypothesis)},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = pipeline(
            prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
        )
        series = pd.DataFrame(test).iloc[ii]
        series['output'] = outputs[0]["generated_text"][len(prompt):]
        series.to_csv(f'{args.output_dir_path}/llama3_nli_{ii}.csv')
        print(outputs[0]["generated_text"][len(prompt):])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="meta-llama/Meta-Llama-3-8B-Instruct", help="pretrained model from HF that can be used for llama prompting")
    parser.add_argument('-out', '--output_dir_path', required=True, help='Directory path where output .csv files will be stored') 
    parser.add_argument('-token', '--hf_token', required=True, help='HuggingFace token') 
    parser.add_argument('-prompt', '--prompt', required=True, help='Prompt method from the available in `experiments/llms/prompts.py`')
    parser.add_argument('-temp', '--temperature', required=False, default=0.6, help='temperature argument for the LLM')
    parser.add_argument('-n_tokens', '--max_new_tokens', type=int, required=False, default=4, help='Maximum number of new tokens (following prompt) to generate')

    args = parser.parse_args()
    main(args)