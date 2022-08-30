import pandas as pd

# Read knowledge triples csv into pandas DataFrame
roc_stories_srl_1_csv = '../generated/srl/ROCStories_resolved_srl1.csv'
roc_stories_srl_1_df = pd.read_csv(roc_stories_srl_1_csv, sep='\t', header=0)
roc_stories_srl_2_csv = '../generated/srl/ROCStories_resolved_srl2.csv'
roc_stories_srl_2_df = pd.read_csv(roc_stories_srl_2_csv, sep='\t', header=0)
roc_stories_srl_3_csv = '../generated/srl/ROCStories_resolved_srl3.csv'
roc_stories_srl_3_df = pd.read_csv(roc_stories_srl_3_csv, sep='\t', header=0)
roc_stories_srl_4_csv = '../generated/srl/ROCStories_resolved_srl4.csv'
roc_stories_srl_4_df = pd.read_csv(roc_stories_srl_4_csv, sep='\t', header=0)
roc_stories_srl_5_csv = '../generated/srl/ROCStories_resolved_srl5.csv'
roc_stories_srl_5_df = pd.read_csv(roc_stories_srl_5_csv, sep='\t', header=0)

# Merge triples into single dataframe
roc_stories_srl_df = roc_stories_srl_1_df
roc_stories_srl_df['srl_r2'] = roc_stories_srl_2_df['srl_r2']
roc_stories_srl_df['srl_r3'] = roc_stories_srl_3_df['srl_r3']
roc_stories_srl_df['srl_r4'] = roc_stories_srl_4_df['srl_r4']
roc_stories_srl_df['srl_r5'] = roc_stories_srl_5_df['srl_r5']

del roc_stories_srl_df['Unnamed: 0']
del roc_stories_srl_df['Unnamed: 0.1']

roc_stories_srl_df.to_csv('../generated/srl/ROCStories_srl.csv', ',')


