import re
from typing import List, Any
import pandas as pd
from tqdm import tqdm
import sys
import os
from allennlp.common import JsonDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import utils.semantic_role_labeling as sem_rl

tqdm.pandas()

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


def srl_to_entities(srl: JsonDict):
    """
    Extract Knowledge triples from semantic roles

    :param srl: PropBank English SRLs
    :return: knowledge triples as a List of Lists
    """
    res = []
    verbs = srl['verbs']
    n_empty = 0
    for d in verbs:
        tags = d['tags']
        triple = d['description']
        verb = d['verb']
        args = re.findall('\\[ARG[01234]+.*?: (.*?)\\]', triple)
        for arg1 in args:
            res.append(arg1)
    if not res:
        print(f"Empty entities for SRL: \n{srl}\n")
    return res


# Read ROCStories into pandas DataFrame
roc_stories_path_csv = '../generated/coreference_resolution/ROCStories_with_resolved_coreferences.csv'
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep='\t', header=0)

srl_entities_df = roc_stories_df

# Add entities to Dataframe for each sentence in the dataset
for n in range(1, 6):
    srl_entities_df[f'srl_entities{n}'] = srl_entities_df[f'resolved{n}'] \
        .progress_apply(lambda s: srl_to_entities(sem_rl.sentence_to_srl(s)))

# Convert Dataframe to csv
srl_entities_df.to_csv('../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities.csv', sep='\t')
