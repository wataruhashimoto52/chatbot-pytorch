# coding: utf-8

import torch
from model import *
from preprocess import *
from global_config import *
from train import *
from loader import *


if __name__ == "__main__":
    
    hidden_size = 256
    input_lang, output_lang, pairs = prepareData('source', 'target', reverse = True)
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words,
        n_layers = 1, dropout_p = 0.1)

    l_encoder, l_decoder, epoch = load_previous_model(encoder1, attn_decoder1, 
                                        CHECKPOINT_DIR, MODEL_PREFIX)
    
    if use_cuda:
        l_encoder = l_encoder.cuda()
        l_decoder = l_decoder.cuda()

    communication(input_lang, output_lang, l_encoder, l_decoder)