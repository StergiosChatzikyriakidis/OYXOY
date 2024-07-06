from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers.util import semantic_search

class FinetunedXNLIBasedICL:
    """ICL for LLM prompting based on 
    embeddings produced by a finetuned model on 
    mnli / xnli data. Model used from HF: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` 
    
    """
    def __init__(self, dev, test, k=10) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        self._get_top_k_samples(dev, test, k=k)
        

    def _get_embeddings(self, data):
        emb = []
        for sample in tqdm(data):
            input = self.tokenizer(sample.premise, sample.hypothesis, truncation=True, return_tensors="pt")
            # forward pass
            emb.append(torch.mean(self.model(**input).hidden_states[-1], dim=1).detach()[0])
        return emb

    def _get_top_k_samples(self, dev, test, k):
        dev_un_emb = self._get_embeddings([sample for sample in dev if 'Unknown' in [label._value_ for label in sample.labels]])
        dev_ent_emb = self._get_embeddings([sample for sample in dev if 'Entailment' in [label._value_ for label in sample.labels]])
        dev_contr_emb = self._get_embeddings([sample for sample in dev if 'Contradiction' in [label._value_ for label in sample.labels]])

        test_emb = self._get_embeddings(test)

        self.res_un = semantic_search(query_embeddings=test_emb, corpus_embeddings=dev_un_emb, top_k=k)
        self.res_ent = semantic_search(query_embeddings=test_emb, corpus_embeddings=dev_ent_emb, top_k=k)
        self.res_contr = semantic_search(query_embeddings=test_emb, corpus_embeddings=dev_contr_emb, top_k=k)

        self.dev_un = [sample for sample in dev if 'Unknown' in [label._value_ for label in sample.labels]]
        self.dev_ent = [sample for sample in dev if 'Entailment' in [label._value_ for label in sample.labels]]
        self.dev_contr = [sample for sample in dev if 'Contradiction' in [label._value_ for label in sample.labels]]

    def get_examples_based_on_finetuned_mnli_xnli(self, n_shots, idx):
        entailment_examples = [self.dev_ent[self.res_ent[idx][ii]['corpus_id']] for ii in range(n_shots//3)]
        unknown_examples = [self.dev_un[self.res_un[idx][ii]['corpus_id']] for ii in range(n_shots//3)]
        contradiction_examples = [self.dev_contr[self.res_contr[idx][ii]['corpus_id']] for ii in range(n_shots//3)]
        return entailment_examples + unknown_examples + contradiction_examples
    

class TagsBasedICL:
    def __init__(self):
        pass
    
    def get_examples_based_on_one_hot_enc(self):
        pass
    