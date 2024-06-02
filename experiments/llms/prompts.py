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

def select_prompt(method):
    if method=='zero_shot_nli_label':
        return zero_shot_nli_label_system, zero_shot_nli_label_user
    elif method=='zero_shot_nli_tags':
        return zero_shot_nli_tags_system