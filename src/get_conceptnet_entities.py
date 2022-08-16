import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json

from nltk.corpus import stopwords


conceptnet_path_csv = '../data/conceptnet-5.7.0/conceptnet.csv'

# Read csv files into pandas Dataframes
conceptnet_df = pd.read_csv(conceptnet_path_csv, sep='\t', header=None,
                            names=['uri', 'relation', 'head', 'tail', 'json_info'])

# Only keep knowledge triples
conceptnet_formatted_df = conceptnet_df[['head', 'relation', 'tail']]
# Only keep english nodes
conceptnet_formatted_english_df = conceptnet_formatted_df[
    conceptnet_formatted_df['head'].str.contains('/en/') & conceptnet_formatted_df['tail'].str.contains('/en/')]

heads = []
for idx, row in tqdm(conceptnet_formatted_english_df.iterrows()):
    heads.append(row['head'].replace('/c/en/', '').replace('/n', '').replace('/v', '').replace('/a', '').replace('/s', '')
                 .replace('/r', '').replace('_', ' '))

entities = set(heads)
stopwords_en = set(stopwords.words('english'))
entities = set([e for e in entities if not e.lower() in stopwords_en])

with open('conceptnet_heads.txt', 'r') as f:
    f.write(json.dumps(entities))
