import pandas as pd

# Read knowledge triples csv into pandas DataFrame
roc_stories_triples_1_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples1.csv'
roc_stories_triples_1_df = pd.read_csv(roc_stories_triples_1_csv, sep='\t', header=0)

roc_stories_triples_2_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples2.csv'
roc_stories_triples_2_df = pd.read_csv(roc_stories_triples_2_csv, sep='\t', header=0)

roc_stories_triples_3_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples3.csv'
roc_stories_triples_3_df = pd.read_csv(roc_stories_triples_3_csv, sep='\t', header=0)

roc_stories_triples_4_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples4.csv'
roc_stories_triples_4_df = pd.read_csv(roc_stories_triples_4_csv, sep='\t', header=0)

roc_stories_triples_5_csv = '../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples5.csv'
roc_stories_triples_5_df = pd.read_csv(roc_stories_triples_5_csv, sep='\t', header=0)

# Merge triples into single dataframe
roc_stories_triples_df = roc_stories_triples_1_df
roc_stories_triples_df['triples2'] = roc_stories_triples_2_df['triples2']
roc_stories_triples_df['triples3'] = roc_stories_triples_3_df['triples3']
roc_stories_triples_df['triples4'] = roc_stories_triples_4_df['triples4']
roc_stories_triples_df['triples5'] = roc_stories_triples_5_df['triples5']

del roc_stories_triples_df['srl_r1']

roc_stories_triples_df.to_csv('../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples.csv',
                              '\t')

# Read entities csv into pandas DataFrame
roc_stories_entities_1_csv = '../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities1.csv'
roc_stories_entities_1_df = pd.read_csv(roc_stories_entities_1_csv, sep='\t', header=0)

roc_stories_entities_2_csv = '../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities2.csv'
roc_stories_entities_2_df = pd.read_csv(roc_stories_entities_2_csv, sep='\t', header=0)

roc_stories_entities_3_csv = '../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities3.csv'
roc_stories_entities_3_df = pd.read_csv(roc_stories_entities_3_csv, sep='\t', header=0)

roc_stories_entities_4_csv = '../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities4.csv'
roc_stories_entities_4_df = pd.read_csv(roc_stories_entities_4_csv, sep='\t', header=0)

roc_stories_entities_5_csv = '../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities5.csv'
roc_stories_entities_5_df = pd.read_csv(roc_stories_entities_5_csv, sep='\t', header=0)

# Merge triples into single dataframe
roc_stories_entities_df = roc_stories_entities_1_df
roc_stories_entities_df['srl_entities2'] = roc_stories_entities_2_df['srl_entities2']
roc_stories_entities_df['srl_entities3'] = roc_stories_entities_3_df['srl_entities3']
roc_stories_entities_df['srl_entities4'] = roc_stories_entities_4_df['srl_entities4']
roc_stories_entities_df['srl_entities5'] = roc_stories_entities_5_df['srl_entities5']

del roc_stories_entities_df['srl_r1']

roc_stories_entities_df.to_csv('../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities.csv',
                              '\t')
