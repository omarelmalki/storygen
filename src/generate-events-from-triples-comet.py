from utils.comet_class import Comet
import pandas as pd
from tqdm import tqdm
import ast

print("model loading ...")
comet = Comet('../lib/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART')
comet.model.zero_grad()
print("model loaded")

roc_stories_triples_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples.csv'
roc_stories_triples_df = pd.read_csv(roc_stories_triples_csv, sep='\t', header=0)

forward_rels = ["xWant", "isBefore", "xEffect"]
backward_rels = ["xIntent", "isAfter", "xNeed"]

result_df = pd.DataFrame()
story_ids = []
sentence_ids = []
results_final = []
for n in range(1, 6):
    print(f"n = {n}")
    tmp_df = pd.DataFrame()
    new_events_rows = []
    queries = []
    relations = []
    triples_res = []

    for index, row in tqdm(roc_stories_triples_df.iterrows()):
        story_id = row['storyid']
        sentence_id = f'sentence_{n}'
        triples = ast.literal_eval(row[f'triples{n}'])
        for idx, triple in enumerate(triples):
            comet_head = ' '.join(triple)
            #             print(comet_head)
            if n < 5:
                for rel in forward_rels:
                    query = "{} {} ".format(comet_head, rel)
                    queries.append(query)
                    relations.append(rel)
                    triples_res.append(triple)
                    # for pivot
                    story_ids.append(story_id)
                    sentence_ids.append(sentence_id)
            if n > 1:
                for rel in backward_rels:
                    query = "{} {} ".format(comet_head, rel)
                    queries.append(query)
                    relations.append(rel)
                    triples_res.append(triple)
                    # for pivot
                    story_ids.append(story_id)
                    sentence_ids.append(sentence_id)

    print('Genrating events...')
    results = comet.generate(queries, decode_method="beam", num_generate=10)
    results = [list(filter(lambda x: x.strip() != 'none', r)) for r in results]
    new_events = [list(a) for a in zip(triples_res, relations, results)]

    results_final += new_events
    print(len(new_events))
    print(len(results_final))
    tmp_df['storyid'] = story_ids
    tmp_df['sentence_id'] = sentence_ids
    tmp_df[f'new_events'] = results_final
    print(f'Saving intermediate result {n}...')
    tmp_df.to_csv(f'../generated/new_events_from_comet/ROCStories_with_new_triples_not_formatted{n}.csv', sep='\t')


result_df['storyid'] = story_ids
result_df['sentence_id'] = sentence_ids
result_df[f'new_events'] = results_final

print(f'Saving not formatted result...')
result_df.to_csv(f'../generated/new_events_from_comet/ROCStories_with_new_triples_not_formatted.csv', sep='\t')

# Reformat file
tmp = result_df.groupby(['storyid', 'sentence_id']).new_events.apply(list).to_frame().reset_index()
res = tmp.pivot(index='storyid', columns='sentence_id', values='new_events').reset_index()
res.columns.name = None
print(f'Saving formatted result...')
res.to_csv(f'../generated/new_events_from_comet/ROCStories_with_new_triples_formatted.csv', sep='\t')
