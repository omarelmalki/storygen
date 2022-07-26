import os
import pandas as pd
from tqdm import tqdm
import ast

tqdm.pandas()

# Read ROCStories into pandas DataFrame
roc_stories_path_csv = "../generated/semantic-role-labeling/ROCStories_resolved_with_knowledge_triples.csv"
roc_stories_df = pd.read_csv(roc_stories_path_csv, sep='\t', header=0)


def triples_to_string(triples):
    """
    Transform list of triples into a single string, omitting the relation
    :param triples: List[List[str]]
    :return: str
    """
    s = ""
    list_triples = ast.literal_eval(triples)
    for triple in list_triples:
        s += triple[0] + " " + triple[2] + " ; "
    return s


roc_stories_triples_df = roc_stories_df[['storyid', 'srl_r1', 'srl_r2', 'srl_r3', 'srl_r4', 'srl_r5']]

for n in range(1, 6):
    roc_stories_triples_df[f'triple_str_t{n}'] = roc_stories_triples_df[f'srl_r{n}'] \
        .progress_apply(lambda t: triples_to_string(t))

roc_stories_string_triples_df = roc_stories_triples_df[
    ['storyid', 'triple_str_t1', 'triple_str_t2', 'triple_str_t3', 'triple_str_t4', 'triple_str_t5']]

roc_stories_string_triples_df.to_csv(
    '../generated/conceptnet-node-extraction/ROCStories_resolved_triples_entity_extraction_input.csv', sep='\t')

os.chdir('../../lib/CoCo-Ex/')
entity_extraction_script_name = 'CoCo-Ex_entity_extraction.py'
input_csv = "../../generated/conceptnet-node-extraction/ROCStories_resolved_triples_entity_extraction_input.csv"
output_tsv = "../../generated/conceptnet-node-extraction/ROCStories_resolved_triples_entity_extraction_output.tsv"
os.system(f'python3 {entity_extraction_script_name} {input_csv} {output_tsv}')

output_filtered_tsv = "../../generated/conceptnet-node-extraction/ROCStories_resolved_triples_conceptnet_nodes_filtered.tsv"
overhead_filter_script_name = 'CoCo-Ex_overhead_filter.py'
len_diff_tokenlevel = 1
len_diff_charlevel = 10
dice_coefficient = 0.8

os.system(f'python3 {overhead_filter_script_name} '
          f'--inputfile {output_tsv} '
          f'--outputfile {output_filtered_tsv} '
          f'--len_diff_tokenlevel {len_diff_tokenlevel} '
          f'--len_diff_charlevel {len_diff_charlevel} '
          f'--dice_coefficient {dice_coefficient}')

roc_stories_extracted_nodes_df = pd.read_csv(output_filtered_tsv, sep='\t', header=None,
                                             names=['storyid', 'sentence_index', 'sentence', 'nodes'])

os.system('cd ../../src/')


def nodes_to_list(node):
    """
    Convert the nodes in a sentence in string format to a set of nodes
    :param node:
    :return:
    """
    return set(node[1:-1].split("]["))


roc_stories_extracted_nodes_df['node_list'] = roc_stories_extracted_nodes_df['nodes']\
    .progress_apply(lambda node: nodes_to_list(node))


