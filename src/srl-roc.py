import pandas as pd
import sys
import os
import json
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
import utils.semantic_role_labeling as sem_rl

# Read ROCStories into pandas DataFrame
roc_stories_path_csv = '../generated/coreference_resolution/ROCStories_with_resolved_coreferences.csv'
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep='\t', header=0)

print(len(roc_stories_df))

predictor = sem_rl.get_predictor()
batch_size = 128
n = sys.argv[1]

batch_data = []
batch_result = []

for index, row in tqdm(roc_stories_df.iterrows()):
    line = str(row[f'resolved{n}'])
    if not line.isspace():
        line = {"sentence": line.strip()}
        line = json.dumps(line)
        json_data = predictor.load_line(line)
        batch_data.append(json_data)
        if len(batch_data) == batch_size:
            res = sem_rl.run_batch_predictor(batch_data, predictor)
            for b in res:
                out = predictor.dump_line(b)
                batch_result.append(out)
            batch_data = []

if batch_data:
    res = sem_rl.run_batch_predictor(batch_data, predictor)
    for b in res:
        out = predictor.dump_line(b)
        batch_result.append(out)

roc_stories_df[f'srl_r{n}'] = batch_result

srl_df = roc_stories_df

# Convert Dataframe to csv
srl_df.to_csv(f'../generated/srl/ROCStories_resolved_srl{n}.csv', sep='\t')
