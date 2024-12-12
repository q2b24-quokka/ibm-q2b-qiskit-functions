#!/usr/bin/python3
'''
Functions to load, process, and combine semnets from SKB-DA
SKB-DA provides source code for the paper "A Data Augmentation Method for Building Sememe Knowledge Base via Reconstructing Dictionary Definitions" by Li and Takano, published in The Association for Natural Language Processing 2022
https://github.com/SauronLee/SKB-DA
'''
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
    
def get_semnet(npy, sort=False):
    dictionary = np.load(npy, allow_pickle=True).tolist()
    print(f'Semnet size: {len(dictionary)}')
    
    if sort:
        keys = list(dictionary.keys())
        keys.sort()
        return {k: dictionary[k] for k in keys}
    else:
        return dictionary

def convert_semnet(semnet):
    new_dictionary = {}
    for clause, sems in semnet.items():
        idxs = []
        for match in re.finditer('\.', clause):
            s, e = match.start(), match.end()
            idxs.append((s,e))

        p_start_end= (idxs[-2][1],idxs[-1][0])
        w_end = idxs[-2][0]

        pos = clause[p_start_end[0]:p_start_end[1]]
        word = clause[:w_end]
        
        underscore = re.search('_', word)
        if not underscore:
            entry = (word, pos)

            if entry in new_dictionary:
                new_dictionary[entry].update(sems)
            else:
                new_dictionary[entry] = sems
    print(f'Converted semnet size: {len(new_dictionary)}')
    return new_dictionary

def convert_semnet_wiki(semnet):
    new_dictionary = {}
    
    for clause, sems in semnet.items():
        word = clause.split('>>>')[0].split(' (')[0]
        underscore = re.search('_', word)
        
        if not underscore:
            tags = pos_tag([word])
            pos = get_wordnet_pos(tags[0][1])

            if pos != None:
                entry = (word, pos)
                if entry in new_dictionary:
                    new_dictionary[entry].update(sems)

                else:
                    new_dictionary[entry] = set(sems)
    print(f'Converted semnet size: {len(new_dictionary)}')
    return new_dictionary

# ATTRIBUTION: https://github.com/SauronLee/SKB-DA/blob/289e7564c834528ff33dd36ecf31c2d65d64a48e/SKB-DA/ad_processing.ipynb
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def merge_semnets(s1, s2):
    s2.update(s1)
    keys = list(s2.keys())
    keys.sort()
    return {k: s2[k] for k in keys}

def meta_semnet():
    # Get Wordnet semnet
    wn5000 = get_semnet('data/sememe_network_dict_en_wordnet_5000.npy')
    wn5000_clean = convert_semnet(wn5000)
    
    # Get Wikipedia semnet
    wk2000 = get_semnet('data/sememe_network_dict_en_wiki_2000.npy')
    wk2000_clean = convert_semnet_wiki(wk2000)
    
    semnet = merge_semnets(wn5000_clean, wk2000_clean)
    print(f'Semnet created with {len(semnet)} words')
    return semnet