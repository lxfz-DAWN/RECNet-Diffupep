from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor




class ESMCembeding(nn.Module):
    def __init__(self, 
                model = ESMC_model,
                device:str='cuda'):
        super(ESMCembeding, self).__init__()
        self.device = device


    def embedding(input_list:list,model = ESMC_model):
        max_length = max([item.sequence.shape[0] for item in input_list])
        padded_merge = []
        for single_merge in input_list:
            current_length = single_merge.sequence.shape[0]
            padding_length = max_length - current_length
            padding = torch.ones((padding_length), device= device, dtype=single_merge.sequence.dtype)
            padded_single_merge = torch.cat([single_merge.sequence.to(device), padding], dim=0)
            single_merge.sequence = padded_single_merge
            padded_merge.append(single_merge)
        all_outputs = []
        for single_padded_merge in padded_merge:
            if acc_or_not == True:
                single_padded_merge = ESMC_model.module.logits(single_padded_merge.to(device),LogitsConfig(sequence = True,return_hidden_states = True))
            else:
                single_padded_merge = ESMC_model.logits(single_padded_merge.to(device),LogitsConfig(sequence = True,return_hidden_states = True))
            single_padded_merge = single_padded_merge.hidden_states[-1, :, :, :]
            all_outputs.append(single_padded_merge)
        all_outputs = torch.stack(all_outputs)
        return all_outputs

    def ESMC_sequence_dealing(seq:str,
                            target_seq:str,
                            Receptor_as_Trimers:bool=True
                            ):
        result = ''
        if Receptor_as_Trimers:
            result = seq + '|' + target_seq
        else:
            result = seq + '|' + target_seq  # ?? 这两段有什么区别？
        return result