import pandas as pd

# Read knowledge triples csv into pandas DataFrame
roc_stories_triples_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples'
roc_stories_triples_df = pd.read_csv(roc_stories_triples_csv, sep='\t', header=0)

# Read knowledge triples csv into pandas DataFrame
roc_stories_cn_nodes_csv = '../generated/conceptnet-node-extraction/ROCStories_resolved_cn_nodes_filtered.csv'
roc_stories_cn_nodes_df = pd.read_csv(roc_stories_cn_nodes_csv, sep='\t', header=0)

res_df = roc_stories_triples_df.merge(roc_stories_cn_nodes_df, how='outer', on='storyid')

