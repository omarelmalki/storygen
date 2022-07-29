###################################################################
# CoCo-Ex V. 1.0
# released by Maria Becker, Katharina Korfhage, and Anette Frank
# please cite our paper when using this software
###################################################################

if __name__ == "__main__":
    print("Importing packages...")
# import re
import os
import re
import csv
import spacy
import pickle
import string
import itertools
# import pandas as pd
import numpy as np
from sys import argv
from glob import iglob
# from itertools import product
from datetime import datetime
from nltk.parse import stanford
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from cos_sim import cos_similarity
from tqdm import tqdm
# from numba import jit, cuda
import multiprocessing as mp
from itertools import chain
from nltk.tree import Tree

# from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
if __name__ == "__main__":
    print("Done.")


class Text:

    def __init__(self, sentences: list):

        self.sents = [self.AnnotatedSentence(sent.strip()) for sent in
                      sentences]  # list of the sentences contained in the input text, represented as Text.AnnotatedSentence objects
        self.sents_without_empty_lines = [sent for sent in self.sents if
                                          sent.text != ""]  # required in order to handle mismatch errors in self.parse

    # @jit(nopython=True, parallel=True)
    def parse(self, parser):

        """
        This will parse Text.sents into constituency parse Trees using the parser defined. 
        A parser could, for example, be the Stanford Parser.
        Input: Parser
        Output: None
        Overwrites parse attribute of Text's AnnotatedSentence objects with their respective parses.
        It is a lot faster to parse as many sentences as possible at once, thus not an individual parse call for each sentence.
        """

        def _run_parse_batch(q, batch_size, sents_wo_empty, idx):
            # result = parser.raw_parse_sents([sent.text for sent in sents_wo_empty])
            batch_data = []
            batch_result = []

            # create a list of constituency parse trees for all sentences that are part of this Text instance (this is faster than individually calling parse for each sentence)

            for sent in tqdm(sents_wo_empty, desc=f'parsing for process: {idx}'):
                txt = sent.text
                batch_data.append(txt)
                if len(batch_data) == batch_size:
                    res = parser.raw_parse_sents(batch_data)
                    batch_result.append(res)
                    batch_data = []

            if batch_data:
                res = parser.raw_parse_sents(batch_data)
                batch_result.append(res)

            q.put((idx, batch_result))

        def _take_first(elem):
            return elem[0]

        n = 3
        size = len(self.sents)
        size_per_process = size // n
        size_first_process = size_per_process + size % n
        parsed_sents = []
        parsed_sents_arr = []

        q = mp.Queue()
        processes = []
        for p in range(n):
            start = 0
            end = size_first_process
            if p > 0:
                start = size_first_process + (p - 1) * size_per_process
                end = start + size_per_process
            sents_wo_empty = self.sents[start:end]
            pr = mp.Process(target=_run_parse_batch, args=(q, 13, sents_wo_empty,p,))
            print(f'starting process: {p}')
            pr.start()
            processes.append(pr)

        for p in range(n):
            print(f'getting result from process: {p}')
            sents_pr = q.get()
            parsed_sents_arr.append(sents_pr)

        for p, pr in enumerate(processes):
            print(f'joining process: {p}')
            pr.join()

        parsed_sents_arr.sort(key=_take_first)
        for ps in parsed_sents_arr:
            for li in ps[1]:
                # for each parse tree created, match the tree with its corresponding sentencde
                for idx, parsed_sent in tqdm(enumerate(li)):
                    parsed_sents.append(parsed_sent)

        # parsed_sents = parser.raw_parse_sents([sent.text for sent in self.sents_without_empty_lines])
        # for each parse tree created, match the tree with its corresponding sentencde
        # print(f'len retrieved: {len(parsed_sents)}')
        # print(parsed_sents)
        # print(f'type retrieved: {type(parsed_sents)}')
        for idx, parsed_sent in tqdm(enumerate(parsed_sents)):
            print(type(parsed_sent))
            self.sents_without_empty_lines[idx].parse = next(
                parsed_sent)  # next is to get the first tree if there are several

        # for idx, p in enumerate(self.sents_without_empty_lines):
        #     if not p.parse:
        #         print(f'index: {idx}, sentence: {p.text}')
        #         # p.parse = Tree()

        return None

    class AnnotatedSentence:

        def __init__(self, sentence: str):

            self.text = sentence  # this is the plain sentence as a string
            self.parse = None  # can be overwritten with constituency parse Tree of self.text (as generated by the Stanford Parser) using Text.parse method
            self.phrases = None  # can be overwritten with list of Phrase objects representing extracted phrases from this sentence's constituency parse, by using self.extract_phrases method.

        def extract_phrases(self, phrase_types: list, remove_pps: bool = False, get_compounds: bool = False):

            """
            This will extract all phrases (trees and subtrees) from the sentence which belong to one of the specified phrase_types.
            Phrases are saved as a list of tuples (phrase, phrase_type) in self.phrases.
            Requires self.parse to be not None, otherwise raises an exception.
            Input:
            * phrase_types: List of all the phrase types to be extracted from the parse tree. Valid phrase types for the English Stanford Parser are all tags from the Penn Treebank tagset.
            * remove_pps: Whether or not to incorporate versions of verb phrases where PP-subtrees are excluded. Enabling this may lead to grammatically incomplete phrases. Verb phrases where a PP was removed are labelled "VP-reduced".
            * get_compounds: Whether or not to incorporate compounds as individual noun phrases. Compounds will be tagged "NP-compound".
            """

            # catch errors when trying to extract phrases from a sentence that has not yet been constituency-parsed (since the parse tree is required for the phrase extraction)
            if not self.parse:
                print(self.text)
                raise Exception(
                    "You tried to extract phrases from a text that has not been constituency-parsed yet. You have to perform a constituency parse first. This can be done for the whole text using the Text class's .parse method.")

            else:
                # set up list of phrases as feature of this AnnotatedSentence instance, to be filled with all the phrases found during extraction
                self.phrases = list()
                # cache the strings of all phrases already seen to be able to avoid duplicates (since no comparison method is implemented for the Text.AnnotatedSentence.Phrase class)
                phrase_strings = set()
                # iteratively search the sentence's parse for phrases of each phrase type requested
                for phrase_type in phrase_types:
                    # go through all subtrees that have the current phrase type as a label
                    for subtree in self.parse.subtrees(filter=lambda x: x.label() == phrase_type):
                        # get the phrase as a string (= the leaves of this subtree of the parse tree)
                        phrase_string = ' '.join(subtree.leaves())
                        # check if this phrase has already been extracted from this string, to avoid duplicates
                        # this is important if checking for compounds (get_compounds) or including reduced verb phrases (exclude_pps)
                        if phrase_string not in phrase_strings:
                            # if the phrase is not a duplicate, add to list of phrases found for this sentence as a Text.AnnotatedSent.Phrase class instance
                            self.phrases.append(self.Phrase(phrase_string, subtree, phrase_type))
                            phrase_strings.add(phrase_string)

                # if versions of verb phrases excluding PPs should be added (remove_pps flag): create artificially reduced phrases where subordinated PP, NP and WHNP are cut from verb phrases
                if remove_pps:
                    # create list that should ultimately contain all reduced verb phrases where subordinate prepositional phrases are removed
                    reduced_phrases = list()
                    # iterate over all phrases extracted from the sentence before, and get the verb phrases out of those
                    for phrase in self.phrases:
                        if phrase.phrase_type == 'VP':
                            # create list containing all subordinate PP, NP and WHNP phrases in this verb phrase
                            subphrases = list()
                            for subtree in phrase.phrase_tree.subtrees(
                                    filter=lambda x: x.label() == 'PP' or x.label() == 'NP' or x.label() == 'WHNP'):
                                subphrases.append(' '.join(subtree.leaves()))
                            # create list of all possible combinations of this verb phrase's subordinate phrases (PP, NP or WHNP)
                            # each of the combination sets in this list will produce one reduced variant of the original verb phrase
                            subphrase_sets = list()
                            for nr_elms in range(1, 2):
                                subphrase_sets.extend([elm for elm in itertools.combinations(subphrases, nr_elms)])
                            # create list of strings of new, reduced verb phrases. For each, some of the subordinate phrases (PP, NP or WHNP) are removed
                            for elm in subphrase_sets:
                                reduced_phrase_string = phrase.phrase_string
                                for substring in elm:
                                    reduced_phrase_string = reduced_phrase_string.replace(' ' + substring, '')
                                reduced_phrases.append(
                                    self.Phrase(reduced_phrase_string, phrase.phrase_tree, 'VP-reduced'))
                    # add all reduced verb phrases to complete list of phrases of the sentence as Phrase objects, where phrase_type is "VP-reduced". Note that tree is still original tree of VP
                    for reduced_phrase in reduced_phrases:
                        if reduced_phrase.phrase_string not in phrase_strings:
                            self.phrases.append(reduced_phrase)
                            phrase_strings.add(reduced_phrase.phrase_string)

                # if get_compounds, get compounds (without other stuff like adjectives) out of NP as individual phrases
                if get_compounds:
                    # set to contain all the compounds found within the sentence
                    compounds = set()
                    # iterate through all the phrases found before, and pick the NPs from those (since only they can contain compounds)
                    for phrase in self.phrases:
                        if phrase.phrase_type == 'NP':
                            # collect all tokens belonging to a compound in a list
                            current_compound = list()
                            # flag to check if a compound is still being continued or if something else came in between
                            # compounds can only be nouns following right after one another, within the same noun phrase!
                            in_compound = False
                            # iterate through the pos tags of all tokens in the NP (as assigned by the stanford parser)
                            for subtree in phrase.phrase_tree.subtrees(filter=lambda x: x.height() == 2):
                                # check if the current token is tagged as a noun
                                if subtree.label() in ['NN', 'NNP', 'NNS', 'NNPS']:
                                    # if yes, mark as part of a compound
                                    if not in_compound:
                                        in_compound = True
                                    current_compound.append(subtree.leaves()[0])
                                # if not, store complete compound if one was found
                                else:
                                    if in_compound:
                                        in_compound = False
                                        if len(current_compound) > 1:
                                            compounds.add(' '.join(current_compound))
                                        current_compound = list()
                            # catch cases where the compound was not yet completed by the end of the last subtree
                            if in_compound:
                                if len(current_compound) > 1:
                                    compounds.add(' '.join(current_compound))
                    # add all compounds that were found to list of phrases of this sentence
                    # note that parse tree will be None and phrase type will be "NP-compound"
                    for compound in compounds:
                        self.phrases.append(self.Phrase(compound, None, 'NP-compound'))

            return None

        class Phrase:

            """
            Object representing an extracted phrase from a sentence's parse tree, has one of a selection of phrase types from a tagset (e.g. the Penn Treebank Tagset for English; phrase types might be: NP, VP, ADJP, NN, JJ, ...).
            VP phrases may be artificially reduced down so that a set of subordinate NP and/or PP are cut off.
            >> In this case, the phrase_tree will still contain the full tree, but the string will not.
            >> If phrases were removed from a VP, the phrase_type will be "VP-reduced".
            Noun compounds can be artificially extracted from NP. 
            >> In this case, the phrase_tree is None and the phrase_type is 'NP-compound'.
            """

            def __init__(self, phrase_string, phrase_tree, phrase_type):

                self.phrase_string = phrase_string  # the string of the extracted phrase
                self.phrase_tree = phrase_tree  # the parse tree of the extracted phrase - for VP-reduced, this is still the original parse tree; for NP-compound, this is None
                self.phrase_type = phrase_type  # the type of the extracted phrase, e.g. "NP" - for VP where PPs or NPs were removed, the type is VP reduced, for noun compounds, it is NP-compound

                self.universal_pos_tagging = None  # will be overwritten by self.preprocess_phrase to contain a list of (token, universal pos-tag) tuples of all tokens in the phrase

                self.tokens = None  # can be overwritten to contain all the tokens of phrase_string as a list, lowercased, by using self.preprocess_phrase method
                self.lemmas = None  # can be overwritten to contain the lemmatized forms of self.tokens as a list, still including stopwords etc, by using self.preprocess_phrase method
                self.normalized = None  # can be overwritten to contain the normalized (lemmatized + stopwords and other stuff removed) form of self.phrase_string/self.tokens, by using self.preprocess_phrase method

                self.candidate_cn_nodes = None  # can be overwritten with a list of all the conceptnet nodes (as ComparableConceptNetNode instances) which contain the phrase's (non-stopword) lemmas, by using self.find_cn_intersection. The nodes can be preprocessed using ComparableConceptNetNode.preprocess_node, and afterwards similarities to the phrase can be calculated using self.calculate_node_similarities

            def preprocess_phrase(self, lemmatizer: spacy.lemmatizer.Lemmatizer, cn_dict: dict,
                                  universal_tagset_mapping: dict, preprocessing_settings: dict,
                                  preprocessed_phrases_cache: dict):

                self.tokens = [token.lower() for token in self.phrase_string.split(" ")]
                self.pos_tagging = list()
                self.lemmas = list()
                self.normalized = list()

                if self.phrase_type == "NP-compound":
                    for token in self.tokens:
                        self.pos_tagging.append((token.lower(), "NOUN"))
                        self.lemmas.append(lemmatizer(token.lower(), "NOUN")[0])

                else:
                    # token_position is to check if the token in the parse tree is actually in the string, because for vp-reduced, the parse tree will contain tokens that are not in the tree
                    token_position = 0
                    for idx, subtree in enumerate(self.phrase_tree.subtrees(filter=lambda x: x.height() == 2)):
                        token = subtree.leaves()[
                            0].lower()  # there should be only one token since trees of height 2 are always POSTAG --> TOKEN
                        universal_tag = universal_tagset_mapping[subtree.label()]
                        # in case the phrase_type is vp-reduced, the phrase tree might contain elements that are not actually in the token list - these should be skipped   
                        if ((idx + 1) > len(self.tokens)) or (token != self.tokens[token_position]):
                            continue
                        else:
                            token_position += 1
                            self.pos_tagging.append((token, universal_tag))
                            if (token, universal_tag) in preprocessed_phrases_cache:
                                self.lemmas.append(preprocessed_phrases_cache[(token, universal_tag)])
                            elif (token == "'s") and (universal_tag == "VERB"):
                                self.lemmas.append("be")
                                preprocessed_phrases_cache[(token, universal_tag)] = "be"
                            else:
                                lemma = lemmatizer(token, universal_tag)[0]
                                # if adjectives are not in cn dict due to the way they are lemmatized (eg, for annoying only annoy is in dict), lemmatize as if it was a verb and use that version of lemma
                                if (lemma not in cn_dict) and (universal_tag == "ADJ"):
                                    lemma = lemmatizer(token, "VERB")[0]
                                self.lemmas.append(lemma)
                                preprocessed_phrases_cache[(token, universal_tag)] = lemma

                for idx, (lemma, pos) in enumerate(zip(self.lemmas, [postag for token, postag in self.pos_tagging])):
                    if lemma in preprocessing_settings["stops"]:
                        continue
                    if (preprocessing_settings["remove_adv"]) and (pos == 'ADV'):
                        continue
                    if (preprocessing_settings["remove_conj"]) and (pos == 'CONJ'):
                        continue
                    if (preprocessing_settings["remove_det"]) and (pos == 'DET'):
                        continue
                    if (preprocessing_settings["remove_intj"]) and (pos == 'INTJ'):
                        continue
                    if (preprocessing_settings["remove_pron"]) and (pos == 'PRON'):
                        continue
                    if (preprocessing_settings["remove_punct"]) and (pos == 'PUNCT'):
                        continue
                    # if no reason to ignore this token in normalization, add to normalized list of tokens
                    self.normalized.append(lemma)

                return preprocessed_phrases_cache

            def find_cn_intersection(self, cn_dict: dict, allow_pos_duplicates: bool = False,
                                     universal_tagset_mapping: dict = dict()):

                # allow_pos_duplicates: there might be duplicates in the conceptnet matches if the node strings are identical but they are tagged with different pos tags in conceptnet. If False, such duplicates will be filtered out

                # make sure that phrase has been normalized before matching with conceptnet
                if self.normalized == None:
                    raise Exception(
                        'You tried to find CN nodes for a phrase which has not been preprocessed: "{}". The phrase must be preprocessed first using the preprocess function!'.format(
                            self.phrase_string))
                # catch cases where normalized phrase is empty (e.g.: only stopwords)
                elif self.normalized == []:
                    self.candidate_cn_nodes = list()
                # get all conceptnet nodes that contain all tokens in the (normalized) phrase
                # this causes a lot of overhead, which has to be filtered later
                else:
                    sets = list()
                    duplicate_checker = dict()
                    # if check_alternative_lemmatizations:
                    for idx, token in enumerate(self.normalized):
                        try:
                            if allow_pos_duplicates:
                                sets.append(cn_dict[token])
                            else:
                                for concept in cn_dict[token]:
                                    concept_string = concept.split("/")[3].replace("_", " ")
                                    if (concept_string not in duplicate_checker) or (
                                            len(duplicate_checker[concept_string]) > len(concept)):
                                        duplicate_checker[concept_string] = concept
                                    sets.append(set(duplicate_checker.values()))
                        except KeyError:
                            pass
                    if len(sets) == 0:
                        self.candidate_cn_nodes = list()
                    else:
                        self.candidate_cn_nodes = [self.ComparableConceptNetNode(node) for node in
                                                   set.intersection(*sets)]

                return None

            def calculate_node_similarities(self,
                                            model=None,
                                            stops: set = set(),
                                            nodes_sim_cache: dict = dict(),
                                            get_exact_matches: bool = True,
                                            get_len_diff: bool = True,
                                            get_dice_sim: bool = True,
                                            get_jaccard_sim: bool = True,
                                            get_wmd: bool = True,
                                            get_med: bool = True,
                                            get_cos_sim: bool = True):

                # calculate the specified similarity metrics for all candidate nodes associated with this phrase
                for i, node in enumerate(self.candidate_cn_nodes):
                    if (node.node, tuple(self.tokens)) in nodes_sim_cache:
                        self.candidate_cn_nodes[i] = nodes_sim_cache[(node.node, tuple(self.tokens))]
                    else:
                        node.calculate_similarities(self.tokens,
                                                    self.lemmas,
                                                    self.normalized,
                                                    model=model,
                                                    stops=stops,
                                                    get_exact_matches=get_exact_matches,
                                                    get_len_diff=get_len_diff,
                                                    get_dice_sim=get_dice_sim,
                                                    get_jaccard_sim=get_jaccard_sim,
                                                    get_wmd=get_wmd,
                                                    get_med=get_med,
                                                    get_cos_sim=get_cos_sim)
                        nodes_sim_cache[(node.node, tuple(self.tokens))] = node

                return nodes_sim_cache

            class ComparableConceptNetNode:

                def __init__(self, node, node_string=None):

                    self.node = node
                    if node_string == None:
                        self.node_string = node.split('/')[3].replace('_', ' ')
                    else:
                        self.node_string = node_string

                    self.tokens = None  # can be overwritten using preprocess_node_new function
                    self.lemmas = None  # can be overwritten using preprocess_node_new function
                    self.normalized = None  # can be overwritten using preprocess_node_new function

                    self.lemmatized_checked = None  # can be overwritten using self.check_lemmatized_node method. Will contain string of self.lemmas IF that is a valid cn node, ELSE None

                    self.exact_match = None  # can be overwritten using self.calculate_similarities. True if all tokens match exactly, else False.
                    self.exact_match_lemmas = None  # can be overwritten using self.calculate_similarities. True if all lemmas match exactly, else False.
                    self.exact_match_nostops = None  # can be overwritten using self.calculate_similarities. True if all lemmas that are not stop words match exactly, else False

                    self.len_diff_token = None  # can be overwritten using self.calculate_similarities. Length difference in nr. of tokens between compared sequences as int, min is 0.
                    self.len_diff_char = None  # can be overwritten using self.calculate_similarities. Length difference in nr. of chars between compared sequences as int, min is 0.

                    self.dice_score = None  # can be overwritten using self.calculate_similarities. Dice coefficient between compared sequences as float.
                    self.dice_score_lemmas = None  # see above, for lemmas
                    self.dice_score_nostops = None  # see above, for lemmas that are not stopwords

                    self.jaccard_score = None  # can be overwritten using self.calculate_similarities. Jaccard similarity between compared sequences as float.
                    self.jaccard_score_lemmas = None  # see above, for lemmas
                    self.jaccard_score_nostops = None  # see above, for lemmas that are not stopwords

                    self.wmd = None  # can be overwritten using self.calculate_similarities. Word mover's distance between compared sequences as float, based on embedding model.
                    self.wmd_lemmas = None  # see above, for lemmas
                    self.wmd_nostops = None  # see above, for lemmas that are not stopwords

                    self.med = None  # can be overwritten using self.calculate_similarities. levenshtein distance between compared sequences as int, min is 0, max is length of longest sequence.
                    self.med_lemmas = None  # see above, for lemmas
                    self.med_nostops = None  # see above, for lemmas that are not stopwords

                    self.cos_sim = None  # can be overwritten using self.calculate_similarities. cosine similarity between compared sequences as float, based on embedding model.
                    self.cos_sim_lemmas = None  # see above, for lemmas
                    self.cos_sim_nostops = None  # see above, for lemmas that are not stopwords

                def check_lemmatized_node(self, cn_dict2: dict):
                    # return True if lemmatized node is a valid cn concept
                    # else False

                    lemmatized = ' '.join(self.lemmas)
                    if lemmatized == self.node_string:
                        self.lemmatized_checked = lemmatized
                    for lemma in self.lemmas:
                        if (lemma in cn_dict2) and (lemmatized in cn_dict2[lemma]):
                            self.lemmatized_checked = lemmatized
                    return None

                def calculate_similarities(self,
                                           compare_phrase_tokens,
                                           compare_phrase_lemmas,
                                           compare_phrase_normalized,
                                           phrase_nodes_sim_cache: dict = dict(),
                                           model=None,
                                           stops: set = set(),
                                           get_exact_matches: bool = True,
                                           get_len_diff: bool = True,
                                           get_dice_sim: bool = True,
                                           get_jaccard_sim: bool = True,
                                           get_wmd: bool = True,
                                           get_med: bool = True,
                                           get_cos_sim: bool = True
                                           ):

                    # catch errors through not loading a model for model-based similarities
                    if (get_wmd or get_cos_sim) and (not model):
                        raise Exception(
                            "You tried to calculate a language model based similarity (Word Mover's Distance or Cosine Similarity) without passing a language model. Please pass a model upon method call.")

                    # compare if the sequences match exactly
                    if get_exact_matches:
                        self.exact_match = compare_phrase_tokens == self.tokens
                        self.exact_match_lemmas = compare_phrase_lemmas == self.lemmas
                        self.exact_match_nostops = compare_phrase_normalized == self.normalized

                    # get difference in length on token/char level
                    if get_len_diff:
                        self.len_diff_token = abs(len(compare_phrase_tokens) - len(self.tokens))
                        self.len_diff_char = abs(len(' '.join(compare_phrase_tokens)) - len(' '.join(self.tokens)))

                    # dice similarities for sequences
                    if get_dice_sim:
                        self.dice_score = dice_coefficient(compare_phrase_tokens, self.tokens)
                        self.dice_score_lemmas = dice_coefficient(compare_phrase_lemmas, self.lemmas)
                        self.dice_score_nostops = dice_coefficient(compare_phrase_normalized, self.normalized)

                    # jaccard similarities for sequences
                    if get_jaccard_sim:
                        self.jaccard_score = jaccard_similarity(compare_phrase_tokens, self.tokens)
                        self.jaccard_score_lemmas = jaccard_similarity(compare_phrase_lemmas, self.lemmas)
                        self.jaccard_score_nostops = jaccard_similarity(compare_phrase_normalized, self.normalized)

                    # word mover's distance of sequences
                    if get_wmd:
                        self.wmd = model.wmdistance(compare_phrase_tokens, self.tokens)
                        self.wmd_lemmas = model.wmdistance(compare_phrase_lemmas, self.lemmas)
                        self.wmd_nostops = model.wmdistance(compare_phrase_normalized, self.normalized)

                    # minimum edit distance of sequences (char level)
                    if get_med:
                        self.med = levenshtein(' '.join(compare_phrase_tokens), self.node_string)
                        self.med_lemmas = levenshtein(' '.join(compare_phrase_lemmas), ' '.join(self.lemmas))
                        self.med_nostops = levenshtein(' '.join(compare_phrase_normalized), ' '.join(self.normalized))

                    # cosine similarity of sequences
                    if get_cos_sim:
                        self.cos_sim = cos_similarity(compare_phrase_tokens, self.tokens, model, stops)
                        self.cos_sim_lemmas = cos_similarity(compare_phrase_lemmas, self.lemmas, model, stops)
                        self.cos_sim_nostops = cos_similarity(compare_phrase_normalized, self.normalized, model, stops)

                    return None


def preprocess_node_new(node_nlp, preprocessing_settings: dict):
    # create preprocessed version of phrase
    # all tokens as list, lowercased
    tokenized = list()
    # all tokens lemmatized
    # catch wrongly lemmatized "be" ("'s") on the go
    lemmatized = list()
    # lemmas of tokens that are not stopwords or to be removed through a flag
    normalized = list()
    # iterate through all tokens of preprocessed phrase
    for idx, token in enumerate(node_nlp):
        # save lowercased token
        tokenized.append(token.lower_)
        # save lemmatized version of token
        lemmatized.append(token.lemma_.lower())
        # check if there is any reason why the current token should not be part of the normalized string
        if token.lemma_ in preprocessing_settings["stops"]:
            continue
        if preprocessing_settings["remove_adv"] and token.pos_ == 'ADV':
            continue
        if preprocessing_settings["remove_conj"] and token.pos_ == 'CONJ':
            continue
        if preprocessing_settings["remove_det"] and token.pos_ == 'DET':
            continue
        if preprocessing_settings["remove_intj"] and token.pos_ == 'INTJ':
            continue
        if preprocessing_settings["remove_pron"] and token.pos_ == 'PRON':
            continue
        if preprocessing_settings["remove_punct"] and token.pos_ == 'PUNCT':
            continue
        # if no reason to ignore this token in normalization, add to normalized list of tokens
        normalized.append(token.lemma_.lower())

    return tokenized, lemmatized, normalized


def jaccard_similarity(sent_1, sent_2):
    # input should be lowercased lists of tokens

    set1 = set(sent_1)
    set2 = set(sent_2)

    return float(len((set1 & set2))) / len((set1 | set2))


def dice_coefficient(sent_1, sent_2):
    """dice coefficient 2nt/na + nb."""
    sent_1_set = set(sent_1)
    sent_2_set = set(sent_2)
    overlap = len(sent_1_set & sent_2_set)

    return overlap * 2.0 / (len(sent_1_set) + len(sent_2_set))


# taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class EntityExtractor:

    def __init__(self, stanford_path: str, cn_dict_path: str, embeddings_path: str, phrases_path: str,
                 universal_tagset_mapping_path: str):

        self.stanford_path = stanford_path
        self.cn_dict_path = cn_dict_path
        self.embeddings_path = embeddings_path
        self.phrases_path = phrases_path
        self.universal_tagset_mapping_path = universal_tagset_mapping_path

        # this is to set up all the stuff that has to be loaded first, so that you can call extract_entities several times later, if you want, without loading everything every time

        print(datetime.now(), "Initializing caches...")
        self.preprocessed_nodes_cache = dict()
        self.phrase_nodes_sim_cache = dict()
        self.preprocessed_phrases_cache = dict()
        self.already_seen_sentences_cache = dict()
        print(datetime.now(), "Done.")

        # configure and load Stanford Parser
        print(datetime.now(), "Setting environment variables for Stanford Parser...")
        os.environ['STANFORD_PARSER'] = self.stanford_path
        os.environ['STANFORD_MODELS'] = self.stanford_path
        os.environ['CLASSPATH'] = self.stanford_path + '/*'
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading Stanford Parser...")
        self.parser = stanford.StanfordParser(
            model_path=self.stanford_path + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading SpaCy model...")
        self.spacy_nlp = spacy.load("en", disable=['parser', 'ner', 'textcat'])
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading SpaCy lemmatizer...")
        self.lemmatizer = self.spacy_nlp.vocab.morphology.lemmatizer
        print(datetime.now(), "Done.")

        print(datetime.now(), "Configuring preprocessing settings...")
        self.stop_words = stopwords.words('english') + ['-PRON-']
        self.preprocessing_settings = {"stops": self.stop_words,
                                       "remove_adv": True,
                                       "remove_conj": True,
                                       "remove_det": True,
                                       "remove_intj": True,
                                       "remove_pron": True,
                                       "remove_punct": True}
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading ConceptNet lookup dictionary...")
        with open(self.cn_dict_path, "rb") as f:
            self.cn_lemmas_dict = pickle.load(f)
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading second ConceptNet lookup dictionary...")
        with open("cn_dict2.p", "rb") as f:
            self.cn_dict2 = pickle.load(f)
        print(datetime.now(), "Done.")

        print(datetime.now(), "Loading embeddings: {}".format(self.embeddings_path))
        # self.model = KeyedVectors.load_word2vec_format(self.embeddings_path, binary=True)
        self.model = None
        print(datetime.now(), "Done.")

        print(datetime.now(), "Getting phrase types to consider as candidate phrases...")
        with open(self.phrases_path) as f:
            self.phrase_types = [line.strip() for line in f]
        print(datetime.now(), "Done.")

        print(datetime.now(), "Getting Penn Treebank to Universal Tagset mapping...")
        with open(self.universal_tagset_mapping_path) as f:
            self.universal_tagset_mapping = dict()
            for line in f:
                if not line.startswith("###"):
                    penn_tag, universal_tag = line.strip().split("\t")
                    self.universal_tagset_mapping[penn_tag] = universal_tag

        return None


def extract_entities(sentences: list, entity_extractor: EntityExtractor):
    # this is the main function of the program which executes the entity extraction

    print(datetime.now(), "Transforming sentences to input format...")
    text = Text(sentences)
    print(datetime.now(), "Done.")

    # create constituency parses of input sentences
    print(datetime.now(), "Applying Stanford Parser to input...")
    text.parse(entity_extractor.parser)
    print(datetime.now(), "Done.")

    nodes_to_preprocess = set()

    print(datetime.now(), "Matching ConceptNet nodes with input sentences...")
    nr_sents = len(text.sents_without_empty_lines)
    for idx, sent in enumerate(text.sents_without_empty_lines):
        print(datetime.now(), "Processing sentence {} of {}...".format(idx + 1, nr_sents))
        # extract phrases from current sentence
        # check if sentence has been seen and cached data can be used for it
        if sent.text in entity_extractor.already_seen_sentences_cache:
            text.sents_without_empty_lines[idx] = entity_extractor.already_seen_sentences_cache[sent.text]
            continue
        print(datetime.now(), "\tExtracting phrases from sentence...")
        sent.extract_phrases(entity_extractor.phrase_types, remove_pps=True, get_compounds=True)
        print(datetime.now(), "\tDone.")
        # loop through all phrases found for sentence
        print(datetime.now(), "\tMatching phrases with candidate nodes...")
        for phrase in sent.phrases:
            # preprocess each phrase
            updated_preprocessed_phrases_cache = phrase.preprocess_phrase(entity_extractor.lemmatizer,
                                                                          entity_extractor.cn_lemmas_dict,
                                                                          entity_extractor.universal_tagset_mapping,
                                                                          entity_extractor.preprocessing_settings,
                                                                          entity_extractor.preprocessed_phrases_cache)
            entity_extractor.preprocessed_phrases_cache.update(updated_preprocessed_phrases_cache)
            # match extracted phrases with candidate ConceptNet nodes
            phrase.find_cn_intersection(entity_extractor.cn_lemmas_dict, allow_pos_duplicates=True,
                                        universal_tagset_mapping=entity_extractor.universal_tagset_mapping)
            # save all nodes that were found for the phrase to preprocess (in a batch, for computation time) later on (unless the entity extractor has their preprocessing cached already)
            for node in phrase.candidate_cn_nodes:
                if node.node_string not in entity_extractor.preprocessed_nodes_cache:
                    nodes_to_preprocess.add(node.node_string)
        print(datetime.now(), "\tDone.")
        entity_extractor.already_seen_sentences_cache[sent.text] = sent
    print(datetime.now(), "Done.")

    print(datetime.now(), "Preprocessing unseen nodes...")
    # preprocess all nodes (from all sentences/phrases in the text, unless they had already been cached) - tokenize, lemmatize and normalize/remove stopwprds
    for node_string, node_nlp in zip(nodes_to_preprocess, entity_extractor.spacy_nlp.pipe(
            [node_string.translate(str.maketrans('', '', string.punctuation)) for node_string in nodes_to_preprocess])):
        entity_extractor.preprocessed_nodes_cache[node_string] = preprocess_node_new(node_nlp,
                                                                                     entity_extractor.preprocessing_settings)
    print(datetime.now(), "Done.")

    print("Removing overhead from candidate nodes...")
    for idx, sent in enumerate(text.sents_without_empty_lines):
        print(datetime.now(), "Processing sentence {} of {}...".format(idx + 1, nr_sents))
        print(datetime.now(), "\tCalculating similarities between sentence's phrases and candidate nodes...")
        for phrase in sent.phrases:
            # loop through all candidate ConceptNet nodes found for the phrase
            for node in phrase.candidate_cn_nodes:
                # cache preprocessed node to save computation time with duplicates
                # update node with preprocessing info from cache
                node.tokens, node.lemmas, node.normalized = entity_extractor.preprocessed_nodes_cache[node.node_string]
                # check if lemmatized form of node is also a valid cn node
                node.check_lemmatized_node(entity_extractor.cn_dict2)
            # process all nodes of a phrase as a batch to improve computation time
            # for all nodes found, calculate their similarity (by specified similarity metrics) to the phrase
            # cache similarities between nodes and phrases once seen, to save computation time with duplicates
            if phrase.phrase_string not in entity_extractor.phrase_nodes_sim_cache:
                entity_extractor.phrase_nodes_sim_cache[phrase.phrase_string] = dict()
            entity_extractor.phrase_nodes_sim_cache[phrase.phrase_string].update(phrase.calculate_node_similarities(
                nodes_sim_cache=entity_extractor.phrase_nodes_sim_cache[phrase.phrase_string],
                model=entity_extractor.model,
                stops=entity_extractor.stop_words,
                get_exact_matches=True,
                get_len_diff=True,
                get_dice_sim=True,
                get_jaccard_sim=False,
                get_wmd=False,
                get_med=False,
                get_cos_sim=False
            ))
        print(datetime.now(), "\tDone.")
    print(datetime.now(), "Done.")

    return text


def write_similarities_file(filename: str, text: Text, ids: list = list()):
    if ids == list():
        ids = list(range(len(text.sents_without_empty_lines)))

    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            "###SENT-ID\tSENT\tPHRASE\tPHRASE-TYPE\tNODE\tNODE-LEMMATIZED\tEXACT-MATCH\tEXACT-MATCH-LEMMAS\tEXACT-MATCH-NOSTOPS\tLEN-DIFF-TOKEN\tLEN-DIFF-CHAR\tDICE\tDICE-LEMMAS\tDICE-NOSTOPS\tJACCARD\tJACCARD-LEMMAS\tJACCARD-NOSTOPS\tWMD\tWMD-LEMMAS\tWMD-NOSTOPS\tMED\tMED-LEMMAS\tMED-NOSTOPS\tCOS\tCOS-LEMMAS\tCOS-NOSTOPS\n")
        for sent_id, sent in zip(ids, text.sents_without_empty_lines):
            for phrase in sent.phrases:
                for node in phrase.candidate_cn_nodes:
                    f.write(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            sent_id,
                            sent.text,
                            phrase.phrase_string,
                            phrase.phrase_type,
                            node.node_string,
                            node.lemmatized_checked,

                            node.exact_match,
                            node.exact_match_lemmas,
                            node.exact_match_nostops,

                            node.len_diff_token,
                            node.len_diff_char,

                            node.dice_score,
                            node.dice_score_lemmas,
                            node.dice_score_nostops,

                            node.jaccard_score,
                            node.jaccard_score_lemmas,
                            node.jaccard_score_nostops,

                            node.wmd,
                            node.wmd_lemmas,
                            node.wmd_nostops,

                            node.med,
                            node.med_lemmas,
                            node.med_nostops,

                            node.cos_sim,
                            node.cos_sim_lemmas,
                            node.cos_sim_nostops
                        ))

    return None


if __name__ == "__main__":

    # takes as input a file where each sentence is on one line
    # will create a list where each sentence is one element, represented as an AnnotatedSentence object

    inputfile = argv[1]
    outputpath = argv[2]
    cn_dict_path = "concepts_en_lemmas.p"
    stanford_path = '/Users/omar/Documents/LIA/tests/storygen/lib/CoCo-Ex/stanford-parser-full-2018-10-17'
    # java_path = '/usr/lib/jvm/java-8-openjdk-amd64/bin/java'
    embeddings_path = 'GoogleNews-vectors-negative300.bin'
    phrases_path = "phrases.txt"
    universal_tagset_mapping_path = "penn_to_universal_tagset_mapping.txt"
    remove_pps = True
    get_compounds = True

    # initialize entity extractor to use with all input texts to come
    entity_extractor = EntityExtractor(stanford_path, cn_dict_path, embeddings_path, phrases_path,
                                       universal_tagset_mapping_path)

    # change this block if your inputfile has a different format!
    print(datetime.now(), "Reading sentences from input files: {}".format(inputfile))
    texts = list()
    c = 0
    for fn in iglob(inputfile):
        with open(fn) as f:
            csv_obj = csv.reader(f, delimiter="\t")
            for row in csv_obj:
                # get sent columns from each row here: the "row" variable is a list of all the columns in a row
                text_id = row[0]
                print(row)
                for sent_id, col in enumerate(row[1:]):
                    print(col)
                    texts.append(("{}_sent{}".format(text_id, sent_id), col.lower()))
    print(datetime.now(), "Done.")
    print(datetime.now(), "Extracting entities...")
    text = extract_entities([sent for sent_id, sent in texts][:], entity_extractor)
    write_similarities_file(outputpath, text, ids=[sent_id for sent_id, sent in texts])
    print(datetime.now(), "Done.")
    print(datetime.now(), "DONE!")
