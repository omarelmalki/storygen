import pandas as pd

# Read ROCStories into pandas DataFrame
roc_stories_2017_path_csv = "../data/rocstories/ROCStories_winter2017.csv"
roc_stories_2016_path_csv = "../data/rocstories/ROCStories_spring2016.csv"
roc_stories_2017_df = pd.read_csv(roc_stories_2017_path_csv, sep=',', header=0)
roc_stories_2016_df = pd.read_csv(roc_stories_2016_path_csv, sep=',', header=0)

roc_stories_df = pd.concat([roc_stories_2016_df, roc_stories_2017_df])

# source_df = roc_stories_df[['storyid', 'sentence1', 'sentence2']]
# target_df = roc_stories_df[['storyid', 'sentence3', 'sentence4', 'sentence5']]
#
# train_source_df = source_df.iloc[:6000]
# train_target_df = target_df.iloc[:6000]
#
# dev_source_df = source_df.iloc[6000:8000]
# dev_target_df = target_df.iloc[6000:8000]
#
# test_source_df = source_df.iloc[8000:10000]
# test_target_df = target_df.iloc[8000:10000]
#
# splits = [['train', train_source_df, train_target_df],
#           ['dev', dev_source_df, dev_target_df],
#           ['test', test_source_df, test_target_df]]
#
# for split in splits:
#     split[1].to_csv(f'../data/storygen/{split[0]}/source.tsv', sep='\t')
#     split[2].to_csv(f'../data/storygen/{split[0]}/target.tsv', sep='\t')

train_df = roc_stories_df.iloc[:6000]
dev_df = roc_stories_df.iloc[6000:8000]
test_df = roc_stories_df.iloc[8000:10000]

train_df.to_csv('../data/storygen/train/train.tsv', sep='\t')
dev_df.to_csv('../data/storygen/dev/dev.tsv', sep='\t')
test_df.to_csv('../data/storygen/test/test.tsv', sep='\t')



