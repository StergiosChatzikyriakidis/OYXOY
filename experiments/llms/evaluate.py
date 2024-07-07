from sklearn.metrics import classification_report
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import re 

def aggregate_results(dir_path):
    csvs = glob(dir_path+'/*/llama3*.csv')
    csvs = sorted(csvs, key= lambda x: int(x.split('.')[0].split('_')[-1]))
    llama_results = []
    for csv_file in tqdm(csvs):
        df = pd.read_csv(csv_file, header=None).T
        df.columns = df.iloc[0]
        df = df[1:]
        llama_results.append(df)
    total = pd.concat(llama_results)
    return total

def extract_integers(sentence):
    """Extracts all integers from a given sentence.

    Args:
        sentence (str): The input sentence from which to extract integers.

    Returns:
        list: A list of integers found in the sentence as strings.
    """
    # Use regular expression to find all integers
    integers = re.findall(r'\b\d+\b', sentence)
    
    return integers
    
def main(args):
    print(args)
    if args.dataset == 'metaphor':
        total = aggregate_results(args.input_dir_path)
        total['output'] = total.output.apply(lambda value: 'True' if value=='yes' else 'False')
        print(classification_report(total['1'], total['output']))

    elif args.dataset == 'word-in-context':
        total = aggregate_results(args.input_dir_path)
        total['output'] = total.output.apply(lambda value: 'True' if value=='yes' else 'False')
        print(classification_report(total['target'], total['output']))

    elif args.dataset == 'sense-selection':
        total = aggregate_results(args.input_dir_path)
        total['output'] = total.output.apply(lambda x: str(int(x)-1) if x.isdigit() else str(int(extract_integers(x)[0])-1) if len(extract_integers(x))>0 else '0')
        print(classification_report(total['def_id'], total['output']))
        
    elif args.dataset == 'inference':
        csvs = glob(args.input_dir_path+'/*/llama3*.csv')
        csvs = sorted(csvs, key= lambda x: int(x.split('.')[0].split('_')[-1]))
        llama_results = []
        for csv_file in tqdm(csvs):
            df = pd.read_csv(csv_file, header=None).T
            df.columns = df.iloc[0]
            df = df[1:]
            llama_results.append(df)
        total = pd.concat(llama_results)
        total.labels = total.labels.apply(lambda labels: labels.strip('{}').split(', ')[0].lower())
        total.output = total.output.apply(lambda out: 'unknown' if out.lower()=='neutral' else out.lower())
        
        print(classification_report(total['labels'], total['output']))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_dir_path', required=True, help='Directory path where output .csv files are stored') 
    parser.add_argument('-prompt', '--prompt', required=True, help='Prompt method from the available in `experiments/llms/prompts.py`')
    parser.add_argument('-ds', '--dataset', type=str, required=False, default=None, help='Dataset on which to run the experiment')

    args = parser.parse_args()
  
    main(args)