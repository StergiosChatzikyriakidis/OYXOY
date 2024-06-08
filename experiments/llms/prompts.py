DATASETS = ['inference', 'sense-selection', 'word-in-context', 'metaphor']
METHODS = {
    'inference': ['zero_shot_nli_label','zero_shot_nli_tags','few_shot_nli_label'],
    'metaphor': ['zero_shot_metaphor'],
    'sense-selection': ['zero_shot_ss'],
    'word-in-context': ['zero_shot_wic'],
}

zero_shot_nli_label_system = "You are an annotator for natural language inference data in greek.\nGiven a premise and a hypothesis, answer with one or two of the words: 'entailment', 'contradiction' or 'neutral'"
zero_shot_nli_label_user = "premise: {}\nhypothesis: {}."

zero_shot_nli_tags_system = """You are an annotator for natural language inference data in greek.\nGiven a premise and a hypothesis, answer only with the tags that should be assigned to the given pair of sentences.
Available Tags:
Linguistic tags are organized hierarchically from least to most specific. When annotating a sample, categorization must proceed all the way down to the most specific entry level available; that is, Logic:Quantification is not a valid tag, because it has children tags Universal, Existential and Non-Standard, whereas Common Sense/Knowledge is valid, seeing as it has no internal subcategorization.

1. Lexical Semantics
    1. Lexical Entailment
        1. Hyponymy
        2. Hypernymy
        3. Synonymy
        4. Antonymy
        5. Meronymy
    2. Morphological Modification 
    3. Factivity
        1. Factive
        2. Non-Factive
    4. Symmetry/Collectivity
    5. Redundancy
    6. FAO
2. Predicate-Argument Structure
    1. Syntactic Ambiguity
    2. Core Arguments
    3. Alternations
    4. Ellipsis
    5. Anaphora/Coreference
    6. Intersectivity
        1. Intersective
        2. Non-Intersective
    7. Restrictivity
        1. Restrictive
        2. Non-Restrictive
3. Logic
    1. Single Negation
    2. Multiple Negations
    3. Conjunction
    4. Disjunction
    5. Conditionals
    6. Negative Concord
    7. Quantification
        1. Universal
        2. Existential
        3. Non-Standard
    8. Comparatives
    9. Temporals
4. Common Sense/Knowledge

"""
zero_shot_nli_label_user = "premise: {}\nhypothesis: {}."

few_shot_nli_label_system = "You are an annotator for natural language inference data in greek.\nGiven a premise and a hypothesis, answer with one or two of the words: 'entailment', 'contradiction' or 'neutral'.\nYou can use the following examples as guidance.\nExamples:\n"
# 1. premise: {entailment_example.premise}\nhypothesis: {entailment_example.hypothesis}\nAnswer: entailment\n2. premise: {unknown_example.premise}\nhypothesis: {unknown_example.hypothesis}\nAnswer: neutral\n3. premise: {contradiction_example.premise}\nhypothesis: {contradiction_example.hypothesis}\nAnswer: contradiction

zero_shot_metaphor_system = "You are a metaphor detection tool that takes as input one sentence responds only with 'yes' if a metaphor appears in the sentence or with 'no' otherwise."

zero_shot_metaphor_user = "Sentence: {}"

zero_shot_wic_system = "You are a word sense disambiguation tool specialized in greek language that takes as input a pair of words (having the same lemma) and a pair of sentences that contain this word and responds only with 'yes' if the word has the same sense in both sentences or with 'no' otherwise."

zero_shot_wic_user = "Words: {}, {}\nSentence 1: {}\nSentence 2: {}"

zero_shot_ss_system = "You are a sense selection tool specialized in greek language that takes as input a word, its definitions and a sentence which contains this word and responds only with the ordering number that corresponds to the correct definition."

zero_shot_ss_user = "Word: {}\nDefinitions: {}\nSentence: {}"

def select_prompt(method, dataset, n_shots=3):
    if dataset == 'inference':
        if method=='zero_shot_nli_label':
            return zero_shot_nli_label_system, zero_shot_nli_label_user
        elif method=='zero_shot_nli_tags':
            return zero_shot_nli_tags_system, zero_shot_nli_label_user
        elif method=='few_shot_nli_label':
            return few_shot_nli_label_system + "\n".join([f"{ii}. "+"premise: {}\nhypothesis: {}\nAnswer: entailment" for ii in range(n_shots)]), zero_shot_nli_label_user
        else:
            assert method in METHODS[dataset], f"Prompt should be one of {METHODS[dataset]}"
    elif dataset == 'metaphor':
        if method=='zero_shot_metaphor':
            return zero_shot_metaphor_system, zero_shot_metaphor_user
        else:
            assert method in METHODS[dataset], f"Prompt should be one of {METHODS[dataset]}"
    elif dataset == 'sense-selection':
        if method == 'zero_shot_ss':
            return zero_shot_ss_system, zero_shot_ss_user
        else:
            assert method in METHODS[dataset], f"Prompt should be one of {METHODS[dataset]}"
    elif dataset == 'sense-selection':
        if method == 'zero_shot_wic':
            return zero_shot_wic_system, zero_shot_wic_user
        else:
            assert method in METHODS[dataset], f"Prompt should be one of {METHODS[dataset]}"
    else:
        assert dataset in DATASETS, f"Dataset should be one of {DATASETS}"