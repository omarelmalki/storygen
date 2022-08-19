import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

tqdm.pandas()

new_events_csv = f'../generated/new_events_from_comet/ROCStories_resolved_new_triples_merged_topk_samples.tsv'
new_events_df = pd.read_csv(new_events_csv, sep='\t', header=0)

sentence_transform = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

new_events_df = new_events_df.iloc[:10]

def add_embeddings(triples):
    if isinstance(triples, float):
        return []

    triples_f = ast.literal_eval(triples)
    for triple in triples_f:
        embeddings = sentence_transform.encode(triple[2])
        triple.append(embeddings)
    return triples_f


for f in tqdm(range(1, 6)):
    new_events_df[f'sent_w_embeddings{f}'] = new_events_df[f'sentence_{f}'].apply(add_embeddings)

new_events_df.to_pickle(f'../generated/new_events_from_comet/ROCStories_with_new_triples_topk_w_embeddings_sample.pk')
