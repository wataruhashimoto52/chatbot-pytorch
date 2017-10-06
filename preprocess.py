# coding: utf-8

from global_config import *
import torch
from torch.autograd import Variable
import MeCab

tagger = MeCab.Tagger("-Owakati")

def japanese_tokenizer(sentence):
    assert type(sentence) is str

    result = tagger.parse(sentence)
    return result.split()
"""
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'SOS', 1:'EOS'}
        self.n_words = 2
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

        else:
            self.word2count[word] += 1
    
    def addSentence(self, sentence):
        for word in japanese_tokenizer(sentence):
            self.addWord(word)

"""


def indexesFromSentence(vocab, sentence):
    """
    Get index number list from sentence.  
    """
    return [vocab[word] for word in japanese_tokenizer(sentence)]

def variableFromSentence(vocab, sentence):
    """
    Get torch.LongTensor variable from sentence.
    """
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_TOKEN)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result
