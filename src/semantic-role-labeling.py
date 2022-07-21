import re
from typing import List, Any
import pandas as pd

from allennlp.common import JsonDict
# Semantic Role Labeling with BERT : https://github.com/Riccorl/transformer-srl
from transformer_srl import predictors


def sentence_to_srl(sentence: str) -> JsonDict:
    """
    Extracts Semantic Roles from a sentence.

    :param sentence: sentence from which to extract semantic roles labels.
    :return: semantic_roles as PropBank English SRLs.
    """

    # Pre-trained model with BERT fine-tuned to predict PropBank SRLs on CoNLL 2012 dataset.
    predictor = predictors.SrlTransformersPredictor.from_path(
        "../data/pre-trained-transformer-srl/srl_bert_base_conll2012.tar.gz", "transformer_srl")

    # More documentation: https://docs.allennlp.org/models/main/models/structured_prediction/predictors/srl/
    semantic_roles: JsonDict = predictor.predict(sentence)
    return semantic_roles


def srl_to_triple(srl: JsonDict) -> List[Any]:
    """
    Extract Knowledge triples from semantic roles

    :param srl: PropBank English SRLs
    :return: knowledge triples as a List of Lists
    """
    res = []
    verbs = srl['verbs']
    print(f'{verbs}\n')
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


# Read ROCStories into pandas DataFrame
roc_stories_path_csv = "../data/rocstories-2017/ROCStories_winter2017.csv"
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep=',', header=0)

srl_df = roc_stories_df

# Add Knowledge triples to Dataframe for each sentence in the dataset
for n in range(1, 6):
    srl_df[f'srl_s{n}'] = srl_df[f'sentence{n}'].apply(lambda s: srl_to_triple(sentence_to_srl(s)))

# Convert Dataframe to csv
srl_df.to_csv('../generated/semantic-role-labeling/ROCStories_with_knowledge_triples', sep='\t')
