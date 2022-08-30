import pandas as pd

# Read ROCStories into pandas DataFrame
roc_stories_2017_path_csv = "../data/rocstories/ROCStories_winter2017.csv"
roc_stories_2016_path_csv = "../data/rocstories/ROCStories_spring2016.csv"
roc_stories_2017_df = pd.read_csv(roc_stories_2017_path_csv, sep=',', header=0)
roc_stories_2016_df = pd.read_csv(roc_stories_2016_path_csv, sep=',', header=0)

roc_stories_df = pd.concat([roc_stories_2016_df, roc_stories_2017_df]).reset_index(drop=True)

roc_stories_df.to_csv('../data/rocstories/ROCStories.csv', sep=',', index=False)
