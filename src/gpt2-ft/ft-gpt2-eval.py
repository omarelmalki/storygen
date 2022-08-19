import pandas as pd
import torch
import os
from tqdm import tqdm
import random
import numpy as np
import time
import datetime
import math
import collections

tqdm.pandas()
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


def build_generator(args, dataset):
    generator = SequenceGenerator(
                    args,
                    Dictionary(dataset.tokenizer.encoder),
                    dataset.tokenizer,
                    beam_size=getattr(args, 'beam', 3),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', dataset.tgt_max_length),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                )
    return generator

def save_generation(args, results, prefix='0'):
    save_result_dir = os.path.join(args.output_dir, "result_ep:{}.txt".format(prefix))
    with open(save_result_dir, 'w') as f:
        for line in results:
            f.write(str(line) + '\n')
    print("Save generation result in {}".format(save_result_dir))


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('train_model')
model = GPT2LMHeadModel.from_pretrained('train_model', pad_token_id=tokenizer.pad_token_id)
# model.resize_token_embeddings(len(tokenizer))

hypos = model.generate(tokenizer.encode("Judy moved into a new home. <|eos|> But soon she felt sick. <|eos|>\nTarget: <|bos|> ", return_tensors='pt'),
                   do_sample=True,
                       max_length=200,
                       top_p=0.8,
                       top_k=50)

print(tokenizer.decode(hypos[0]))

# model = torch.nn.DataParallel(model)

model = model.to(device)


def remove_padding(string, padding='<|pad|>'):
    return string.split(sep=padding)[0]




class StorygenTestDataset(Dataset):
    def __init__(self, tokenizer, data_path, data_type, max_length=1024):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.bos = "<|bos|>"
        self.pad = "<|pad|>"
        self.eos = "<|eos|>"

        self.src_input_ids = []
        self.attn_masks = []
        self.targets = []

        input_sentences_df = pd.read_csv(data_path, sep='\t', header=0)
        input_sentences_df = input_sentences_df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]

        for idx, src in tqdm(input_sentences_df.iterrows()):
            input_str = self.bos
            for s_id in range(1, 3):
                input_str += ' ' + src[f'sentence{s_id}'] + self.eos
            input_str += '\nTarget:'
            if data_type != 'test':
                for s_id in range(3, 6):
                    input_str += ' ' + src[f'sentence{s_id}'] + self.eos
            else:
               target_str = ''
               for s_id in range(3, 6):
                   target_str += ' ' + src[f'sentence{s_id}'] + self.eos
               self.targets.append(target_str)


            encodings_dict = self.tokenizer(input_str, truncation=True, max_length=max_length, padding="max_length")
            self.src_input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.src_input_ids)

    def __getitem__(self, item):
        return self.src_input_ids[item], self.attn_masks[item]


test_dataset = StorygenTestDataset(tokenizer, '../../data/storygen/test/test.tsv', 'test')

batch_size = 1

test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler=SequentialSampler(test_dataset), # Select batches randomly
            batch_size=batch_size # Trains with this batch size.
        )

gen_seqs = []
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):

    batch = tuple(t.to(device) for t in batch)

    print(tokenizer.decode(batch[0][0]))

    with torch.no_grad():

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        # print(tokenizer.decode(b_input_ids[0], skip_special_tokens=True))
        # print(len(b_input_ids))
        # print(remove_padding(tokenizer.decode(b_input_ids[0])))
        b_input = tokenizer.encode(remove_padding(tokenizer.decode(b_input_ids[0])), return_tensors='pt')
        inputs = b_input_ids
        # print(inputs)
        print(remove_padding(tokenizer.decode(b_input_ids[0])))
        hypos = model.generate(b_input,
            do_sample=True,
            max_length=200,
            top_p=0.8,
            top_k=50)
        print(hypos)
        generated = tokenizer.decode(hypos[0], skip_special_tokens=True)
        print(generated)
        gen_seqs.append(hypos)
