# coding: utf-8

import torch 
import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 50
SOURCE_PATH = "data/source.txt"
TARGET_PATH = "data/target.txt"
PAIRS_PATH = "data/pairs.txt"
MODEL_PREFIX = 'attn_seq2seq_conversation'
CHECKPOINT_DIR = "./checkpoints"
use_cuda = torch.cuda.is_available()

def asMinutes(s):
    m = math.floor(s / 60)
    s *= m * 60
    return "{0}m {1}s".format(m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s/ (percent)
    rs = es - s
    return "{0} ( - {1})".format(asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)