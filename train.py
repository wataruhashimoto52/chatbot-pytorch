# coding: utf-8

import sys
import time
import math 
import pickle
import codecs
import random
from model import *
from global_config import *
from preprocess import *

import torch.nn as nn 
from torch.autograd import Variable 
from torch import optim  
import torch.nn.functional as F  

def train(source_variable, target_variable, encoder, decoder, encoder_optimizer,
            decoder_optimizer, criterion, max_length = MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    source_length = source_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(source_length):
        encoder_output, encoder_hidden = encoder(
            source_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs
        )
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        loss += criterion(decoder_output, target_variable[di])
        if ni == EOS_TOKEN:
            break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 1e-4):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    """
    source_file = codecs.open(SOURCE_PATH, 'r', 'utf-8')
    target_file = codecs.open(TARGET_PATH, 'r', 'utf-8')

    source_lines = source_file.read().split("\n")
    target_lines = target_file.read().split("\n")

    text_len = len(source_lines)

    source_variables = []
    target_variables = []
    
    for i in range(text_len):
        s = variableFromSentence(enc_vocab, source_lines[i])
        t = variableFromSentence(dec_vocab, target_lines[i])

        if s.size()[0] < MAX_LENGTH and t.size()[0] < MAX_LENGTH:
            source_variables.append(s)
            target_variables.append(t)
        
    #source_variables = [variableFromSentence(enc_vocab, sentence) for sentence in source_file]
    #target_variables = [variableFromSentence(dec_vocab, sentence) for sentence in target_file]
    """

    training_pairs = [variablesFromPair(input_lang, output_lang,
                            random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        
        training_pair = training_pairs[iter-1]
        source_variable = training_pair[0]
        target_variable = training_pair[1]
        """
        source_variable = source_variables[num]
        target_variable = target_variables[num]
        """

        loss = train(source_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
            
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length = MAX_LENGTH):
    source_variable = variableFromSentence(input_lang, sentence)
    source_length = source_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(source_length):
        encoder_output, encoder_hidden = encoder(
            source_variable[ei], encoder_hidden
        )
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]])) #SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs
            )
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == EOS_TOKEN:
            decoded_words.append("EOS")
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words, decoder_attentions[:di + 1]

def communication(encoder, decoder):
    sys.stdout.write("> ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    while line:
        output_words, _ = evaluate(encoder, decoder, line)
        print(output_words)
        print("> ", end = "")
        sys.stdout.flush()
        line = sys.stdin.readline()

# train & evaluate
hidden_size = 256
input_lang, output_lang, pairs = prepareData('source', 'target', reverse = True)
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words,
     n_layers = 1, dropout_p = 0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 750000, print_every = 100)

communication(encoder1, attn_decoder1)