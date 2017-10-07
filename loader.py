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
         # this step using for delete the more than 5 and litter one
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
        for f in to_delete:
            os.remove(f)
    name = model_prefix + '-{}.pth'.format(epoch)
    file_path = os.path.join(checkpoint_dir, name)
    model_dict = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }
    torch.save(model_dict, file_path)
