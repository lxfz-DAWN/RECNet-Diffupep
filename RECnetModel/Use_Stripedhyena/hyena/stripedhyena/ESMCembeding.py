from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def ESMC_600M_202412(device: torch.device | str = "cuda", use_flash_attn: bool = True, model_path: str = None):
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    if model_path is not None:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        # 去除 "module." 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    else:
        print(f"Error: The model file at {model_path} was not found.")
    return model

class ESMCembeding(nn.Module):
    def __init__(self, model, acc_or_not, device:str='cuda'):
        super(ESMCembeding, self).__init__()
        self.device = device
        self.model = model
        self.acc_or_not = acc_or_not


    def embed(self, input:list|Tensor):

        if isinstance(input,list):
            input_list = input
            max_length = max([item.sequence.shape[0] for item in input_list])
            padded_merge = []
            for single_merge in input_list:
                current_length = single_merge.sequence.shape[0]
                padding_length = max_length - current_length
                padding = torch.ones((padding_length), device= self.device, dtype=single_merge.sequence.dtype)
                padded_single_merge = torch.cat([single_merge.sequence.to(self.device), padding], dim=0)
                single_merge.sequence = padded_single_merge
                padded_merge.append(single_merge)
            all_outputs = []
            for single_padded_merge in padded_merge:
                if self.acc_or_not == True:
                    single_padded_merge = self.model.module.logits(single_padded_merge.to(self.device),LogitsConfig(sequence = True,return_hidden_states = True))
                else:
                    single_padded_merge = self.model.logits(single_padded_merge.to(self.device),LogitsConfig(sequence = True,return_hidden_states = True))
                single_padded_merge = single_padded_merge.hidden_states[-1, :, :, :]
                all_outputs.append(single_padded_merge)
            all_outputs = torch.stack(all_outputs)
        else:
            single_padded_merge = input
            single_padded_merge = self.model.logits(single_padded_merge.to(self.device),LogitsConfig(sequence = True,return_hidden_states = True))
            single_padded_merge = single_padded_merge.hidden_states[-1, :, :, :]
            all_outputs = single_padded_merge
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
    
    def unembed(self, u:Tensor) -> Tensor:
        return 1

class ESMCtokenizer():
    def __init__(self, acc_or_not, model):
        self.acc_or_not = acc_or_not
        self.model = model

    def tokenize(self, sequence:str, *args, **kwargs):
        protein = ESMProtein(sequence = sequence)
        protein_tensor = self.model.encode(protein)
        return protein_tensor