import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

tqdm.pandas()

forward_rels = ["xWant", "isBefore", "xEffect"]
backward_rels = ["xIntent", "isAfter", "xNeed"]
sentence_transform = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def forward_filter(triple):
    return triple[1] in forward_rels


def backward_filter(triple):
    return triple[1] in backward_rels


def get_forward_triples(triples, index):
    forward_events = []
    for triple in triples:
        if triple[1] == forward_rels[index]:
            forward_events.append(triple)
    return forward_events


def get_backward_events(triples, index):
    backward_events = []
    for triple in triples:
        if triple[1] == backward_rels[index]:
            backward_events += triple[2]
    return [a.strip() for a in backward_events]


def get_backward_embeddings(triples, index):
    backward_embeddings = np.zeros((0,384))
    print(backward_embeddings.shape)
    for triple in triples:
        if triple[1] == backward_rels[index]:
            if triple[3].size > 0:
                backward_embeddings = np.append(backward_embeddings,triple[3],axis=0)
    return np.array(backward_embeddings)


def backward_similarity(forward_event, backward_array):
    for b in backward_array:
        if util.pytorch_cos_sim(forward_event.astype(np.float64), b.astype(np.float64)) > 0.8:
            return True
    return False


def keep_top_k_elements(l, k):
    if len(l) < k:
        return set(l)
    else:
        return set(l[:k])


new_events_df = pd.read_pickle(f'../generated/new_events_from_comet/ROCStories_with_new_triples_topk_w_embeddings.pkl')

n_rels = len(forward_rels)
pruned_triples = [[] for _ in range(5)]
deleted_triples = [[] for _ in range(5)]

for idx, row in tqdm(new_events_df.iterrows()):
    print(f'row {idx}')
    for f in range(1, 5):
        print(f'row {idx}, sentence {f}')
        triples_f = row[f'sent_w_embeddings{f}']
        forward_events = list(filter(forward_filter, triples_f))
        backward_events = []
        for b in range(f + 1, 6):
            triples_b = row[f'sent_w_embeddings{b}']
            backward_events += list(filter(backward_filter, triples_b))
        pruned_triples_f = []
        for rel in range(n_rels):
            print(f'row {idx}, sentence {f}, relation {forward_rels[rel]}')
            ft = get_forward_triples(forward_events, rel)
            b_embeddings = get_backward_embeddings(backward_events, rel)
            new_ft = []
            for triple in ft:
                triple_copy = [a for a in triple]
                triple_copy[2] = keep_top_k_elements([event.strip() for event, embedding in zip(triple[2], triple[3]) if
                                                      backward_similarity(embedding, b_embeddings)], 5)
                if triple_copy[2]:
                    new_ft.append(triple_copy[:3])
            pruned_triples_f += new_ft
        pruned_triples[f - 1].append(pruned_triples_f)

for f in range(1, 5):
    new_events_df[f'pruned_comet{f}'] = pruned_triples[f - 1]

for f in range(1, 6):
    del new_events_df[f'sent_w_embeddings{f}']

for f in range(1, 6):
    new_events_df.rename({f'sentence_{f}': f'new_events{f}'})

new_events_df.to_csv(f'../generated/new_events_from_comet/ROCStories_with_new_triples_topk_pruned.csv', sep='\t')
