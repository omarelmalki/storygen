import pandas as pd
import spacy
import neuralcoref
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

# Read ROCStories into pandas DataFrame
roc_stories_2017_path_csv = "../data/rocstories/ROCStories_winter2017.csv"
roc_stories_2016_path_csv = "../data/rocstories/ROCStories_spring2016.csv"
roc_stories_2017_df = pd.read_csv(roc_stories_2017_path_csv, sep=',', header=0)
roc_stories_2016_df = pd.read_csv(roc_stories_2016_path_csv, sep=',', header=0)

roc_stories_df = pd.concat([roc_stories_2016_df, roc_stories_2017_df])


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
    """
    Return resolved sentence for sentence indexed by index in story with coreference resolution result coref_res

    :param coref_res: coreference resolution result of story
    :param index: index of sentence in story
    :return: resolved sentence or "" if resolution failed
    """
    n = len(coref_res[1])
    if n < 5:
        print(f"Resolved sentence list has only {n} elements:\n{coref_res}\n")
        # ideally, to be handled manually
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

# Manually add resolved sentences for the few stories with failing coreferences
story_ids = ['08f13294-29ea-412b-ad0d-e05ff9662003', '1382f6a2-459f-417d-9d7e-83e6d0e02e74', '83ab16e0-6f0f-4c0b-94f4-2d5e192d1bbd', '7f8aa034-57b6-4720-85cf-6a4d1da43bc3', '80e24095-17dd-4175-b8e7-1d790bdce7c0']

story1_resolved = ['the minivan parked in the middle of the driveway in our building.', 'The driver loaded some luggage.', 'The driver kept the minivan in the driveway in our building.', 'Another car came behind the minivan and waited.', 'After the driver beeped, the minivan drove off.']
story2_resolved = ['The delivery man handed a box to Gen.', "Gen was excited since Gen's new shoes were inside of a box.", "Once Gen tried Gen's new shoes on, Gen couldn't make Gen's new shoes fit.", "Gen spent hours trying to fix it but made no improvements.", "As a result, Gen returned Gen's new shoes the next day."]
story3_resolved = ['Annabelle laid lifelessly against the window.', 'Annabelle was awaiting the arrival of her owner.', "The sound of the garage door opening alerted Annabelle's ear to perk.", 'Annabelle arose and ran towards the door.', 'Annabelle found her owner standing there with a dog treat in hand.']
story4_resolved = ['Tina was waiting in line at the ice cream shop.', 'The woman in front of Tina was having a lot of samples.', 'This irritated Tina.', 'So Tina decided to confront The woman in front of Tina about it.', 'The woman in front of Tina got mad but did apologize.']
story5_resolved = ['Sally had never made a salad before but thought it would be easy.', 'Sally washed the lettuce, tomatoes, and cucumber well.', 'Sally chopped up the tomato and cucumber.', 'The lettuce Sally tore by hand and put the lettuce, tomatoes, and cucumber all together in a large bowl.', 'Sally forked out a good portion and covered a good portion with ranch dressing.']

resolved_stories = [story1_resolved, story2_resolved, story3_resolved, story4_resolved, story5_resolved]

for n in range(1, 6):
    for i in range(1, 6):
        coref_df.loc[coref_df['storyid'] == story_ids[n-1], f'resolved{i}'] = resolved_stories[n-1][i-1]

# Convert Dataframe to csv
coref_df.to_csv('../generated/coreference_resolution/ROCStories_with_resolved_coreferences.csv', sep='\t')
