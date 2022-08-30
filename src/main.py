import os

coreference_resolution_script = 'coreference-resolution.py'
srl_script = 'srl-roc.py'
srl_merging_script = 'merge-srls.py'
triples_extraction_script = 'knowledge-triple-extraction.py'
triples_merging_script = 'merge-sentences.py'
conceptnet_entities_script = 'get_conceptnet_entities.py'
cn_entities_extraction_script = 'extract-conceptnet-entities.py'

# BEAM
new_events_comet_beam_script = 'generate-events-from-triples-comet.py'
pruning_script_beam = 'prune_new_triples.py'

#TOP-K
new_events_comet_topk_prefix = 'generate-events-from-triples-topk-sample'
merge_new_events_comet_topk_script = 'merge_splitted_result.py'
save_sentence_embeddings_script = 'save_sentence_embeddings.py'
pruning_script_topk = 'prune_new_triples.py'

os.system(f'python {coreference_resolution_script}')

for n in range(1, 6):
    os.system(f'python {srl_script} {n}')

os.system(f'python {srl_merging_script}')

for n in range(1, 6):
    os.system(f'python {triples_extraction_script} {n}')

os.system(f'python {triples_merging_script}')

os.system(f'python {conceptnet_entities_script}')
os.system(f'python {cn_entities_extraction_script}')

# BEAM
# os.system(f'python {new_events_comet_beam_script}')
# os.system(f'python {pruning_script_beam}')

# TOP-K
os.chdir('cluster_specific_scripts/')

for n in range(4):
    os.system(f'python {new_events_comet_topk_prefix}-{n}.py')

os.system(f'python {save_sentence_embeddings_script}')
os.system(f'python {pruning_script_topk}')










