from tqdm import tqdm
import transformers
import torch
import pandas as pd
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.llms.datasets import PromptHandlerSelector
import argparse
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
    PromptHandlerSelector(args.dataset, args.prompt, args.num_shots)
    
    prompt_handler = PromptHandlerSelector(args.dataset).create(args.prompt, args.num_shots)
        
    Path(args.output_dir_path).mkdir(parents=True, exist_ok=True)

    for test_set_name, test_set in prompt_handler.test_sets.items():
        for ii, sample in tqdm(enumerate(test_set)):

            messages = prompt_handler.get_messages(sample, ii)

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
            series = pd.DataFrame(test_set).iloc[ii]
            series['output'] = outputs[0]["generated_text"][len(prompt):]
            series.to_csv(f'{args.output_dir_path}/{test_set_name}/llama3_{ii}.csv')
            print(outputs[0]["generated_text"][len(prompt):])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="meta-llama/Meta-Llama-3-8B-Instruct", help="pretrained model from HF that can be used for llama prompting")
    parser.add_argument('-out', '--output_dir_path', required=True, help='Directory path where output .csv files will be stored') 
    parser.add_argument('-token', '--hf_token', required=True, help='HuggingFace token') 
    parser.add_argument('-prompt', '--prompt', required=True, help='Prompt method from the available in `experiments/llms/prompts.py`')
    parser.add_argument('-temp', '--temperature', required=False, default=0.6, help='temperature argument for the LLM')
    parser.add_argument('-n_tokens', '--max_new_tokens', type=int, required=False, default=4, help='Maximum number of new tokens (following prompt) to generate')
    parser.add_argument('-n_shots', '--num_shots', type=int, required=False, default=None, help='Number of examples to use for the ICL')
    parser.add_argument('-ds', '--dataset', type=int, required=False, default=None, help='Dataset on which to run the experiment')

    args = parser.parse_args()

    assert 'few_shot' in args.prompt and args.num_shots is not None, "You selected a few-shot prompt so you should provide an integer value for `-n_shots` "
    main(args)
