import pandas as pd
import os
from tqdm import tqdm
import ast

tqdm.pandas()

os.chdir('../../src')


def entities_to_string(entities):
    """
    Transform list of entities into a single string
    :param entities:
    :return: str
    """
    s1 = entities.encode('utf-8')
    s1 = s1.decode('utf-8')
    s = ""
    l = ast.literal_eval(s1)
    for entity in l:
        s += entity + ", "
    return s


def nodes_to_set(node):
    """
    Convert the nodes in a sentence in string format to a set of nodes
    :param node:
    :return:
    """
    if not isinstance(node, str):
        return set()
    return set(node[1:-1].split("]["))


# Read ROCStories into pandas DataFramed∫
roc_stories_path_csv = "../generated/prop-bank-entity-extraction/ROCStories_resolved_with_entities.csv"
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep='\t', header=0)

n_df = 10
size = len(roc_stories_df)
size_per_df = size//n_df
size_first_df = size_per_df + size % n_df

for k in range(n_df):
    start = 0
    end = size_first_df
    if k > 0:
        start = size_first_df + (k-1)*size_per_df
        end = start + size_per_df

    roc_stories_n_df = roc_stories_df[start:end]

    roc_stories_entities_df = roc_stories_n_df[['storyid', 'srl_entities1', 'srl_entities2',
                                              'srl_entities3', 'srl_entities4', 'srl_entities5']]

    # Get entity string per sentence from entity list
    for n in range(1, 6):
        roc_stories_entities_df[f'entities_str_t{n}'] = roc_stories_entities_df[f'srl_entities{n}'] \
            .progress_apply(lambda t: entities_to_string(t))

    roc_stories_entities_string_df = roc_stories_entities_df[
        ['storyid', 'entities_str_t1', 'entities_str_t2', 'entities_str_t3', 'entities_str_t4', 'entities_str_t5']]

    # Save input of ConceptNet Entity Extraction script (CoCo-EX)
    roc_stories_entities_string_df.to_csv(
        f'../generated/conceptnet-node-extraction/ROCStories_resolved_entities_entity_extraction_input{k}.csv',
        sep='\t', header=None, index=False)

    # Run CoCo-Ex entity-extraction script
    os.chdir('../lib/CoCo-Ex/')
    entity_extraction_script_name = 'CoCo-Ex_entity_extraction.py'
    input_csv = f"../../generated/conceptnet-node-extraction/ROCStories_resolved_entities_entity_extraction_input{k}.csv"
    output_tsv = f"../../generated/conceptnet-node-extraction/ROCStories_resolved_entities_entity_extraction_output{k}.tsv"
    os.system(f'python3 {entity_extraction_script_name} {input_csv} {output_tsv}')

    # Run CoCo-Ex overhead filter script
    output_filtered_tsv = f"../../generated/conceptnet-node-extraction/RddOCStories_resolved_entities_conceptnet_nodes_filtered{k}.tsv"
    overhead_filter_script_name = 'CoCo-Ex_overhead_filter.py'
    len_diff_tokenlevel = 1
    len_diff_charlevel = 10
    dice_coefficient = 0.8

    # Result is saved in $overhead_filter_script_name
    os.system(f'python3 {overhead_filter_script_name} '
              f'--inputfile {output_tsv} '
              f'--outputfile {output_filtered_tsv} '
              f'--len_diff_tokenlevel {len_diff_tokenlevel} '
              f'--len_diff_charlevel {len_diff_charlevel} '
              f'--dice_coefficient {dice_coefficient}')

    roc_stories_extracted_nodes_df = pd.read_csv(output_filtered_tsv, sep='\t', header=None,
                                                 names=['storyid', 'sentence_index', 'sentence', 'nodes'])

    roc_stories_extracted_nodes_df['node_set'] = roc_stories_extracted_nodes_df['nodes']\
        .progress_apply(lambda node: nodes_to_set(node))
    del roc_stories_extracted_nodes_df['nodes']

    os.chdir('../../src/')

    formatted_df = roc_stories_extracted_nodes_df.pivot(index='storyid', columns='sentence_index', values='node_set')\
        .reset_index()

    formatted_df.columns.name = None
    formatted_df.columns = ['storyid', 'cn_nodes1', 'cn_nodes2', 'cn_nodes3', 'cn_nodes4', 'cn_nodes5']

    # Convert Dataframe to csv
    formatted_df.to_csv(f'../generated/conceptnet-node-extraction/ROCStories_resolved_cn_nodes_filtered{k}.tsv', sep='\t')
