import pandas as pd

new_events_part0_csv = f'../../generated/new_events_from_comet/ROCStories_with_new_triples_formatted_part0_topk.csv'
new_events_part0_df = pd.read_csv(new_events_part0_csv, sep='\t', header=0)

new_events_part1_csv = f'../../generated/new_events_from_comet/ROCStories_with_new_triples_formatted_part1_topk.csv'
new_events_part1_df = pd.read_csv(new_events_part1_csv, sep='\t', header=0)

new_events_part2_csv = f'../../generated/new_events_from_comet/ROCStories_with_new_triples_formatted_part2_topk.csv'
new_events_part2_df = pd.read_csv(new_events_part2_csv, sep='\t', header=0)

new_events_part3_csv = f'../../generated/new_events_from_comet/ROCStories_with_new_triples_formatted_part3_topk.csv'
new_events_part3_df = pd.read_csv(new_events_part3_csv, sep='\t', header=0)

result = pd.concat([new_events_part0_df, new_events_part1_df, new_events_part2_df, new_events_part3_df])

result.to_csv(f'../../generated/new_events_from_comet/ROCStories_with_new_triples_formatted_topk_sample.csv', sep='\t')

roc_stories_triples_csv = '../../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples.csv'
roc_stories_triples_df = pd.read_csv(roc_stories_triples_csv, sep='\t', header=0)

print(len(result))

res_df = roc_stories_triples_df.iloc[:10000].merge(result, how='outer', on='storyid')

print(len(res_df))

del res_df['srl_r1']
del res_df['Unnamed: 0_x']
del res_df['Unnamed: 0.1']
del res_df['Unnamed: 0.1.1']
del res_df['Unnamed: 0.1.1.1']
del res_df['Unnamed: 0_y']

res_df.to_csv('../../generated/new_events_from_comet/ROCStories_resolved_new_triples_merged_topk_samples.tsv', '\t')
