import pandas as pd
import spacy
import neuralcoref
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

# Read ROCStories into pandas DataFrame
roc_stories_path_csv = "../data/rocstories-2017/ROCStories_winter2017.csv"
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep=',', header=0)


def resolve_story(*args: str):
    """
    Return coreference clusters and resolved list of sentences from a list oof input sentences

    :param args: input sentences
    :return: coreference clusters
    :return: resolved sentences
    """
    story = ""
    n = len(args)

    for i in range(n):
        story += args[i]
        if i != n - 1:
            story += "\t"

    coref_res = nlp(story)
    result = [x for x in coref_res._.coref_resolved.split("\t")]
    return coref_res._.coref_clusters, result


def coref_res_to_resolved_sentence(coref_res, index):
    n = len(coref_res[1])
    if n < 5:
        print(f"Resolved sentence list has only {n} elements:\n{coref_res}\n")
        return ""
    if index > 5:
        raise IndexError(f"Index {index} out of bounds for stories of 5 sentences")
    else:
        return coref_res[1][index-1]


coref_df = roc_stories_df

# Apply resolution to all rows
print("Apply resolution to all rows:\n")
coref_df['coref_result'] = coref_df.progress_apply(
    lambda row: resolve_story(row.sentence1, row.sentence2, row.sentence3, row.sentence4, row.sentence5), axis=1)

# Add coreference clusters to dataframe
print("Add coreference clusters to dataframe:\n")
coref_df['coref_clusters'] = coref_df['coref_result'].progress_apply(lambda x: x[0])

# Add Resolved sentence to Dataframe for each sentence in the dataset
print("Add Resolved sentence to Dataframe for each sentence in the dataset:\n")
for i in range(1, 6):
    coref_df[f'resolved{i}'] = coref_df['coref_result'].progress_apply(lambda x: coref_res_to_resolved_sentence(x, i))
del coref_df['coref_result']

# Convert Dataframe to csv
coref_df.to_csv('../generated/coreference_resolution/ROCStories_with_resolved_coreferences', sep='\t')
