import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Read ROCStories into pandas DataFrame
roc_stories_path_csv = "../data/rocstories/ROCStories.csv"
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep=',', header=0)

roc_stories_df['id'] = roc_stories_df['storyid']

roc_stories_df['source'] = roc_stories_df['sentence1'] + ' ' + roc_stories_df['sentence2']

roc_stories_df['target'] = roc_stories_df['sentence3'] + ' ' + roc_stories_df['sentence4'] + ' ' + roc_stories_df['sentence5']

formatted_data = roc_stories_df[['id', 'source', 'target']].to_dict(orient='records')

print(len(formatted_data))

random_seed = 42
train, rem = train_test_split(formatted_data, train_size=0.9, random_state=random_seed)

val, test = train_test_split(rem, test_size=0.5, random_state=random_seed)

print(len(train), len(val), len(test))

with open("../data/storygen/t5_ROCStories_train.json", "w") as fp:
    json.dump(train, fp)

with open("../data/storygen/t5_ROCStories_val.json", "w") as fp:
    json.dump(val, fp)

with open("../data/storygen/t5_ROCStories_test.json", "w") as fp:
    json.dump(test, fp)
