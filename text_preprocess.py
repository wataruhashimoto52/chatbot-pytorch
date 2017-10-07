# coding: utf-8

import re
import codecs
import pickle
from global_config import *

#For Japanese tokenizer
import MeCab

tagger = MeCab.Tagger("-Owakati")

SOURCE_PATH = "data/source.txt"
TARGET_PATH = "data/target.txt"
PAIRS_PATH = "data/pairs.txt"
_WORD_SPLIT = re.compile("([.,!/?\":;)(])")

def japanese_tokenizer(sentence):
    assert type(sentence) is str

    result = tagger.parse(sentence)
    return result.split()

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words]

if __name__ == "__main__":
    
    with open(SOURCE_PATH) as s, open(TARGET_PATH) as t, \
        open(PAIRS_PATH, "w") as p:
        p.write(''.join([c1.strip("\n") + "\t" + c2 for c1, c2 in zip(s, t)]))
    
    # tokenize and idnize
    enc_vocab = {'SOS':0, 'EOS':1}
    enc_lines = codecs.open(SOURCE_PATH, "r", "utf-8")
    for enc_line in enc_lines:
        enc = japanese_tokenizer(enc_line)
        for w in enc:
            if w not in enc_vocab:
                enc_vocab[w] = len(enc_vocab)

    dec_vocab = {'SOS':0, 'EOS':1}
    dec_lines = codecs.open(TARGET_PATH, "r", "utf-8")
    for dec_line in dec_lines:
        dec = japanese_tokenizer(dec_line)
        for w in dec:
            if w not in dec_vocab:
                dec_vocab[w] = len(dec_vocab)

    # serialize 
    with open("enc_vocab.pickle", "wb") as f:
        pickle.dump(enc_vocab, f)

    with open("dec_vocab.pickle", "wb") as d:
        pickle.dump(dec_vocab, d)