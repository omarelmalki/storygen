import pandas as pd
import ast
from tqdm import tqdm

tqdm.pandas()

new_events_csv = f'../generated/new_events_from_comet/ROCStories_resolved_new_triples_merged_topk_samples.tsv'
new_events_df = pd.read_csv(new_events_csv, sep='\t', header=0)

forward_rels = ["xWant", "isBefore", "xEffect"]
backward_rels = ["xIntent", "isAfter", "xNeed"]


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


n_rels = len(forward_rels)
pruned_triples = [[] for _ in range(5)]
for idx, row in tqdm(new_events_df.iterrows()):
    print(f'row {idx}')
    for f in range(1, 5):
        print(f'row {idx}, sentence {f}')
        triples_f = row[f'sentence_{f}']
        if isinstance(triples_f, float):
            pruned_triples[f-1].append([])
            continue
        triples_f = ast.literal_eval(triples_f)
        forward_events = list(filter(forward_filter, triples_f))
        backward_events = []
        for b in range(f+1, 6):
            triples_b = row[f'sentence_{b}']
            if not isinstance(triples_b, float):
                triples_b = ast.literal_eval(triples_b)
                backward_events += list(filter(backward_filter, triples_b))
        pruned_triples_f = []
        for rel in range(n_rels):
            print(f'row {idx}, sentence {f}, relation {forward_rels[rel]}')
            ft = get_forward_triples(forward_events, rel)
            be = get_backward_events(backward_events, rel)
            new_ft = []
            for triple in ft:
                triple_copy = [a for a in triple]
                triple_copy[2] = set([event.strip() for event in triple[2] if event.strip() in be])
                if triple_copy[2]:
                    new_ft.append(triple_copy)
            pruned_triples_f += new_ft
        pruned_triples[f-1].append(pruned_triples_f)

for f in range(1, 5):
    new_events_df[f'pruned_comet{f}'] = pruned_triples[f-1]

for f in range(1, 6):
    new_events_df.rename({f'sentence_{f}': f'new_events{f}'})

new_events_df.to_csv(f'../generated/new_events_from_comet/ROCStories_with_new_triples_topk_pruned_sample.csv', sep='\t')
