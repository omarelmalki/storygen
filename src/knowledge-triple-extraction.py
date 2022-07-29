import re
from typing import List, Any
from allennlp.common import JsonDict
import pandas as pd
# from pandarallel import pandarallel
from tqdm import tqdm
import sys
import os
import json

tqdm.pandas()

# pandarallel.initialize(progress_bar=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import utils.semantic_role_labeling as sem_rl


def srl_to_triple(srl) -> List[Any]:
    """
    Extract Knowledge triples from semantic roles

    :param srl: PropBank English SRLs
    :return: knowledge triples as a List of Lists
    """
    srl = json.loads(srl)
    res = []
    verbs = srl['verbs']
    for d in verbs:
        tags = d['tags']
        triple = d['description']
        verb = d['verb']
        if 'B-ARG0' in tags:
            arg0 = re.search('\\[ARG0: (.*?)\\]', triple).group(1)
            args = re.findall('\\[ARG[1234M]+.*?: (.*?)\\]', triple)
            for arg1 in args:
                res.append([arg0, verb, arg1])
        elif 'B-ARG1' in tags:
            arg1 = re.search('\\[ARG1: (.*?)\\]', triple).group(1)
            args = re.findall('\\[ARG[234M]+.*?: (.*?)\\]', triple)
            for arg2 in args:
                res.append([arg1, verb, arg2])
    if not res:
        print(f"Empty triples for SRL: \n{srl}\n")
    return res


def sentence_to_triple(s):
    res = srl_to_triple(sem_rl.sentence_to_srl(s))
    return res


n = sys.argv[1]

# Read ROCStories into pandas DataFrame
roc_stories_path_csv = f'../generated/srl/ROCStories_resolved_srl{n}.csv'
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep='\t', header=0)

srl_df = roc_stories_df

# Add Knowledge triples to Dataframe for each sentence in the dataset
srl_df[f'triples{n}'] = srl_df[f'srl_r{n}'].progress_apply(srl_to_triple)

# Convert Dataframe to csv
srl_df.to_csv(f'../generated/knowledge-triple-extraction/ROCStories_resolved_with_knowledge_triples{n}.csv', sep='\t')
