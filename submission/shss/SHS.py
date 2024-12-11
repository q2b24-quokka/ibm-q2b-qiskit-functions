#!/usr/bin/python3

import re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from shss.semnet import *

# constants
WORD_PATTERN = r'[^a-zA-Z\s]'
STOP_WORDS_PATTERN = r'\b(i|me|my|myself|we|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now|c|f)\b'

class Corpus:
    wnl = WordNetLemmatizer()
    
    def __init__(self, sentences, semnet):
        self.entries = set() # { ('word', 'pos'), ('word', 'pos'), ... }
        self.docs = []  # [ [ ('word', 'pos'), ('word', 'pos'), ...], ... ]
        self._process(sentences, semnet)

    def _process(self, sentences, semnet):
        for s in sentences:
            entries = list(self._validate_tokens(self._normalize(s), semnet))
            self.docs.append(entries)
    
    def _validate_tokens(self, tokens, semnet):
        entries = set()
        
        for t in tokens:
            tags = pos_tag([t])
            pos = get_wordnet_pos(tags[0][1])

            if pos != None:
                entry = (t, pos)
                if entry in semnet:
                    self.entries.add(entry)
                    entries.add(entry)
        return entries
    
    def _normalize(self, string):
        string = re.sub(WORD_PATTERN, '', string)
        string = re.sub(STOP_WORDS_PATTERN, ' ', string)
        tokens = [Corpus.wnl.lemmatize(token) for token in string.lower().split()]
        return tokens

class SemanticHilbertSpace:
    def __init__(self, corpus, semnet):
        self.num_sememes = 0
        self.sem_to_val = {} # { 'sememe_string': idx, ... }
        self.word_to_sem = self._get_wts(corpus.entries, semnet)    # { (word, pos): {'sem', 'sem' ...} ... }
        self.corpus = self._get_entangled_sememes(corpus.docs)

    def _get_wts(self, entries, semnet):
        m = defaultdict(set)
        curr_sid = 0
        
        for entry in entries:
            raw_sememes = semnet[entry]
            norm_sememes = [re.sub(WORD_PATTERN, '', string) for string in raw_sememes]
            sememes = list(filter(lambda s: len(s) > 0, norm_sememes))
            m[entry] = sememes
            for sem in sememes:
                if sem not in self.sem_to_val:
                    self.sem_to_val[sem] = curr_sid
                    curr_sid += 1
        self.num_sememes = curr_sid
        return m

    def _get_entangled_sememes(self, corpus):
        corpus_entangled = []
        for doc in corpus:
            sem_states = set()
            for entry in doc:
                sems = self.word_to_sem[entry]
                for sem in sems:
                    sem_val = self.sem_to_val[sem]
                    sem_states.add(sem_val)
            corpus_entangled.append(sem_states)
        return corpus_entangled


class SHSQuery:
    wnl = WordNetLemmatizer()
    
    def __init__(self, query, shs, semnet):
        self.sememes = set()
        self.query = query
        self.state = self._get_entangled(shs, semnet)

    def _get_entangled(self, shs, semnet):
        state = set()
        query = self._normalize(self.query)
        
        for token in query:
            tags = pos_tag([token])
            pos = get_wordnet_pos(tags[0][1])
            if pos != None:
                entry = (token, pos)
                if entry in semnet:
                    sememes = semnet[entry]
                    # print(entry, sememes)
                    for sem in sememes:
                        if sem in shs.sem_to_val:
                            self.sememes.add(sem)
                            sem_val = shs.sem_to_val[sem]
                            state.add(sem_val)
        return state
    
    def _normalize(self, string):
        string = re.sub(WORD_PATTERN, '', string)
        string = re.sub(STOP_WORDS_PATTERN, ' ', string)
        tokens = [SHSQuery.wnl.lemmatize(token) for token in string.lower().split()]
        return tokens