from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from transformers import EsmTokenizer, EsmModel, Trainer, TrainingArguments
# from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import default_collate

ESMC_model = ESMC.from_pretrained('esmc_600m').to('cuda')
def ESMC_encoding(sequence:str):
    '''
    调用ESMC进行Encoding，但是不进行计算，用途是输出forward方法可以接受的张量
    '''
    protein = ESMProtein(sequence = sequence)
    protein_tensor = ESMC_model.encode(protein)
    return protein_tensor

seq1 = 'ASDF'
seq2 = '!@#|a'
a = ESMC_encoding(seq1).to('cuda')
b = ESMC_encoding(seq2).to('cuda')
list_test = [a,b]
print(list_test)

def ESMC_embedding(input_list:list,model = ESMC_model):
    max_length = max([item.sequence.shape[0] for item in input_list])
    padded_merge = []
    for single_merge in input_list:
        current_length = single_merge.sequence.shape[0]
        padding_length = max_length - current_length
        padding = torch.ones((padding_length), device='cuda', dtype=single_merge.sequence.dtype)
        padded_single_merge = torch.cat([single_merge.sequence, padding], dim=0)
        single_merge.sequence = padded_single_merge
        padded_merge.append(single_merge)
    all_outputs = []
    for single_padded_merge in padded_merge:
        single_padded_merge = ESMC_model.logits(single_padded_merge,LogitsConfig(sequence = True,return_hidden_states = True))
        single_padded_merge = single_padded_merge.hidden_states[-1, :, :, :]
        all_outputs.append(single_padded_merge)
    all_outputs = torch.stack(all_outputs)
    return all_outputs

test = ESMC_embedding(list_test)
print(test)