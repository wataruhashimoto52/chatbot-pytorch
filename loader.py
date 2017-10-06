# coding: utf-8

import torch
import os
import glob
import numpy as np  

def load_previous_model(encoder, decoder, checkpoint_dir, model_prefix):
    pass

def save_model(encoder, decoder, checkpoint_dir, model_prefix, epoch, max_keep = 5):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    f_list = glob.glob(os.path.join(checkpoint_dir, model_prefix) + '-*.pth')

    if len(f_list) >=max_keep + 2:
        pass