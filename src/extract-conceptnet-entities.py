import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import ast
import json
import spacy

nlp = spacy.load('en_core_web_sm')

with open('conceptnet_heads.txt', 'r') as f:
    entities = set(json.loads(f.read()))

roc_stories_triples_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples.csv'
roc_stories_triples_df = pd.read_csv(roc_stories_triples_csv, sep='\t', header=0)
del roc_stories_triples_df['srl_r1']


def triple_to_entity_set(triples):
    triples = ast.literal_eval(triples)
    s = []
    for triple in triples:
        s.append(triple[0].strip())
        s.append(triple[2].strip())
    return set(s)


def triple_to_verb_set(triples):
    triples = ast.literal_eval(triples)
    s = [triple[1].strip() for triple in triples]
    return set(s)


def lemmatize_set(input_set):
    res = []
    for e in input_set:
        doc = nlp(e)
        res.append(" ".join([token.lemma_ for token in doc]))
    return set(res)


def extract_cn_nodes(input_set):
    res = []
    for e in input_set:
        words = e.lower().split()
        two_grams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        words += two_grams
        res += [e.strip() for e in words if e.strip() in entities]
    return set(res)


def extract_cn_nodes_from_triples(triples):
    verb_set = triple_to_verb_set(triples)
    entity_set = triple_to_entity_set(triples)
    lemmatized_verb_set = lemmatize_set(verb_set)
    lemmatized_entity_set = lemmatize_set(entity_set)

    verb_cn_nodes = extract_cn_nodes(verb_set)
    entity_cn_nodes = extract_cn_nodes(entity_set)
    lemmatized_verb_cn_nodes = extract_cn_nodes(lemmatized_verb_set)
    lemmatized_entity_cn_nodes = extract_cn_nodes(lemmatized_entity_set)

    cn_nodes = {}
    cn_nodes['verb'] = verb_cn_nodes
    cn_nodes['entity'] = entity_cn_nodes
    cn_nodes['verb_lemma'] = lemmatized_verb_cn_nodes
    cn_nodes['entity_lemma'] = lemmatized_entity_cn_nodes

    return cn_nodes


for n in range(1, 6):
    roc_stories_triples_df[f'cn_nodes{n}'] = roc_stories_triples_df[f'triples{n}']\
        .progress_apply(extract_cn_nodes_from_triples)

roc_stories_triples_df.to_csv('../generated/conceptnet-node-extraction/ROCStories_with_cn_heads.csv',
                              sep='\t')
