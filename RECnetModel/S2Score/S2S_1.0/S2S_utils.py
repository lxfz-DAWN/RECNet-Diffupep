# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# import torch
# import torch.nn as nn
# import pandas as pd
# import torch
# import csv
# from transformers import EsmTokenizer, EsmModel, Trainer, TrainingArguments, get_linear_schedule_with_warmup
# from S2S_model import SelfAttention_Module, CrossAttention_Module, Regression_Module
# import torch.optim as optim
# import logging
# import os
# from accelerate import Accelerator
# from transformers import DefaultDataCollator
# from torch.utils.data import DataLoader

# 注意：20250217 对RosettaScore改成了纯MSE，检查loss计算中的问题。

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import torch.nn as nn
import pandas as pd
import csv
from transformers import Trainer, get_linear_schedule_with_warmup
from S2S_model import SelfAttention_Module, CrossAttention_Module, Regression_Module
import torch.optim as optim
import logging
import os
from accelerate import Accelerator
from transformers import DefaultDataCollator,TrainingArguments
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import csv

client = ESMC.from_pretrained("esmc_600m").to('cuda') # or "cpu"

def ESMC_func(sequence:str):
   '''
   To Use ESMC to get the hidden states of the input sequence
   In this model, the ESMC model is modified, the padding length is fixed to 256
   
   Input:
   sequence: str
   '''
   protein = ESMProtein(sequence= sequence)
   protein_tensor = client.encode(protein)
   logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
   
   return logits_output.hidden_states.to('cpu')


def ESMC_saving(input_csv:str, 
                output_csv:str, 
                pt_dir:str,
                ligand_col_name:str = 'ligand_sequence',
                receptor_col_name:str ='receptor_sequence',
                mode:str = 'merge'):
    '''
    Reading a csv file, extract ligand and receptor sequences, transform them into a ESMC embedding tensor and save it.
    *** Rely on function ESMC_func to get the embedding tensor. ***
    *** Must create dirs ***
    
    Input:
        input_csv:str, 
        output_csv:str, 
        pt_dir:str,
        ligand_col_name:str = 'ligand_sequence',
        receptor_col_name:str ='receptor_sequence',
        mode:str = 'merge' (or 'individual')
    '''
    # 这一段代码可以读取seq，添加|后在一个文件夹保存pt文件用于保存ESMC输出。
    # 配置日志信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 读取训练数据
    train_df = pd.read_csv(input_csv)
    # 用于保存张量的目录
    embedding_dir = pt_dir
    # 打开一个文件对象用于写入CSV
    output_file = output_csv
    
    if mode == 'merge':
        total_rows = len(train_df)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            header = train_df.columns.tolist() + ['ESMC_Embedding_Path']
            writer.writerow(header)

            # 逐行处理数据
            for index, row in train_df.iterrows():
                try:
                    ligand_seq = row[ligand_col_name]
                    receptor_seq = row[receptor_col_name]
                    if isinstance(ligand_seq, str) and isinstance(receptor_seq, str):
                        combined_seq = ligand_seq + '|' + receptor_seq + '|' + receptor_seq + '|' + receptor_seq
                    if isinstance(ligand_seq, dict) and isinstance(receptor_seq, dict):
                        combined_seq = '|'.join(str(val) for val in ligand_seq.values()) + '|' + '|'.join(str(val) for val in receptor_seq.values())
                        # row[ligand_col_name] = '|'.join(str(val) for val in ligand_seq.values())
                        # row[receptor_col_name] = '|'.join(str(val) for val in receptor_seq.values())
                    
                    if len(combined_seq) <= 510:
                        embedding = ESMC_func(combined_seq)
                        # 保存张量到文件
                        
                        embedding_dir_exists = os.path.join(embedding_dir,'merge')
                        if not os.path.exists(embedding_dir_exists):
                            os.makedirs(embedding_dir_exists)
                        embedding_path = os.path.join(embedding_dir,'merge',f"embedding_{index}.pt")
                        
                        torch.save(embedding, embedding_path)

                        # 写入当前行数据和嵌入信息文件路径
                        row_data = row.tolist() + [embedding_path]
                        writer.writerow(row_data)

                        # 打印处理进度
                        progress = (index + 1) / total_rows * 100
                        logging.info(f"Processed row {index + 1}/{total_rows} ({progress:.2f}%)")

                        # 释放GPU缓存
                        torch.cuda.empty_cache()
                    else:
                        continue
                except Exception as e:
                    logging.error(f"Error processing row {index}: {e}")
                    
    if mode == 'split':
        total_rows = len(train_df)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            header = train_df.columns.tolist() + ['ESMC_Ligand_Path'] + ['ESMC_Receptor_Path']
            writer.writerow(header)

            # 逐行处理数据
            for index, row in train_df.iterrows():
                try:
                    ligand_seq = row[ligand_col_name]
                    receptor_seq = row[receptor_col_name]
                    if isinstance(ligand_seq, dict) and isinstance(receptor_seq, dict):
                        ligand_seq = '|'.join(str(val) for val in ligand_seq.values())
                        receptor_seq = '|'.join(str(val) for val in receptor_seq.values())
                        
                    if len(ligand_seq) <= 510 and len(receptor_seq) <= 510:
                        embedding_ligand = ESMC_func(ligand_seq)
                        embedding_receptor = ESMC_func(receptor_seq)

                        # 保存张量到文件
                        embedding_ligand_path = os.path.join(embedding_dir,'ligand', f"embedding_{index}.pt")
                        embedding_receptor_path = os.path.join(embedding_dir,'receptor', f"embedding_{index}.pt")
                        embedding_ligand_path_exists = os.path.join(embedding_dir,'ligand')
                        embedding_receptor_path_exists = os.path.join(embedding_dir,'receptor')
                        if not os.path.exists(embedding_ligand_path_exists):
                            os.makedirs(embedding_ligand_path_exists)
                        if not os.path.exists(embedding_receptor_path_exists):
                            os.makedirs(embedding_receptor_path_exists)
                        torch.save(embedding_ligand, embedding_ligand_path)
                        torch.save(embedding_receptor, embedding_receptor_path)

                        row_data = row.tolist() + [embedding_ligand_path] + [embedding_receptor_path]
                        writer.writerow(row_data)
                        
                        # 打印处理进度
                        progress = (index + 1) / total_rows * 100
                        logging.info(f"Processed row {index + 1}/{total_rows} ({progress:.2f}%)")

                        # 释放GPU缓存
                        torch.cuda.empty_cache()
                    else:
                        continue
                except Exception as e:
                    logging.error(f"Error processing row {index}: {e}")

class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df, 
                 score_columns, 
                 merge_hidden,
                 ligand_hidden,
                 receptor_hidden):
        """
        Initialize the database.
        
        Input:
        df: total datacsv, as pd
        score_columns:list of regression targets, list
        merge_hidden:path of merge_tensor, name of colums in df
        ligand_hidden:path of ligand_tensor, name of colums in df
        receptor_hidden:path of receptor_tensor, name of colums in df
        mode = 'foward':default is foward, only gives 'ligand','receptor' and 'merge'. If 'loss', only gives 'label'
        
        """
        self.df = df
        self.score_columns = score_columns
        self.merge_hidden = merge_hidden
        self.ligand_hidden = ligand_hidden
        self.receptor_hidden = receptor_hidden

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 数据索引
        :return: 包含输入特征和回归目标的字典
        """

        # 读取 ESMC 隐藏层特征文件路径
        
        path = self.df.loc[idx, self.merge_hidden]
        merge = torch.load(path, weights_only=True).to('cpu')
        merge = merge[-1, :, :, :]
        path = self.df.loc[idx, self.ligand_hidden]
        ligand = torch.load(path, weights_only=True).to('cpu')
        ligand = ligand[-1, :, :, :]
        path = self.df.loc[idx, self.receptor_hidden]
        receptor = torch.load(path, weights_only=True).to('cpu')
        receptor = receptor[-1, :, :, :]
        scores = self.df.loc[idx, self.score_columns].values
        scores = scores.astype(float)
        scores = torch.tensor(scores, dtype=torch.float).to('cpu')
        data = {'ligand': ligand,'receptor':receptor,'merge':merge,'labels': scores}
        
        return data

class GradNormLoss(nn.Module):
    '''
        Gradnorm, balance the gradient of different tasks.
        
        Input:
        num of task:int
        alpha:float, default is 1.5
        
        GradNormLoss.additional_foward_and_backward(grad_norm_weights:nn.Module, optimizer:optim.Optimizer)
        
    '''
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None

    # standard forward pass
    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.w.device != L_t.device:
            self.w.data = self.w.data.to(L_t.device)
            
        if self.L_0 is None:
            self.L_0 = L_t.detach() # detach
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights: nn.Module, 
            optimizer: optim.Optimizer):
        # do `optimizer.zero_grad()` outside
        self.total_loss.backward(retain_graph=True)
        # in standard backward pass, `w` does not require grad
        self.w.grad.data = self.w.grad.data * 0.0

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(
                self.L_t[i], 
                grad_norm_weights.parameters(), # 这个参数的功能是指定哪一些层的参数会被计算
                retain_graph=True, 
                create_graph=True
                )
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t) # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        optimizer.step()

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task
    

# This is AN interface.
# class GradNormModel:
#     def get_grad_norm_weights(self) -> nn.Module:
#         raise NotImplementedError(
#             "Please implement the method `get_grad_norm_weights`")

class S2S_Loss():
    '''
    这个函数应该有的功能：
        应对不同的cut形式
        单独训练和加和训练两种模式
    '''
    def __init__(self, 
                 label_cut:list, #[62,1,1,1] # 前62个rosettascore和三个特殊变量
                 train_target:list, # ['RosettaScore','kd','TF','IC50'],大小写区分清楚，只有kd小写
                 ):
        self.label_cut = label_cut
        self.train_target = train_target
        self.mse_loss_fn = nn.MSELoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        # 为RosettaScore的62个维度引入独立的不确定性参数
        self.log_var_rosetta = nn.Parameter(torch.tensor([-0.5]* 62))
        
        self.csv_file = open('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/debug/20250217_why_so_big/2.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_row = []

    def cut(self, model_output: torch.Tensor):
        """
        根据 label_cut 对模型输出进行裁剪
        """
        cut_outputs = []
        start = 0
        for cut in self.label_cut:
            end = start + cut
            cut_outputs.append(model_output[:, start:end])
            start = end
        # cut_output = [[batch_size, 62], [batch_size, 1], [batch_size, 1], [batch_size, 1]] (if label_cut = [62, 1, 1, 1])
        return cut_outputs

    def MSE_loss(self, output, target):
        """
        计算 MSE 损失
        """
        # print(f'target{target},output{output},mse{self.mse_loss_fn(output, target)}')
        return self.mse_loss_fn(output, target)

    def BCE_loss(self, output, target):
        """
        计算二元交叉熵损失
        """
        if output.shape[1] == 2:
            # 这里假设 target 为 0 或 1，取对应位置的概率
            probabilities = output.gather(1, target.long().unsqueeze(1)).squeeze(1)
            # 计算 BCE 损失
            return self.bce_loss_fn(probabilities, target.squeeze(1))
        
        return self.bce_loss_fn(output, target)

    def uncertainty_mse_loss(self, output, target):
        """
        计算引入不确定性的MSE损失
        """
        # log_var_rosetta = self.log_var_rosetta.to(output.device)
        # precision = torch.exp(-log_var_rosetta)
        # # 分别计算每个维度的MSE损失
        # mse_losses = torch.mean((output - target) ** 2, dim=0)
        # # 引入不确定性参数
        # losses = precision * mse_losses + log_var_rosetta
        # # 对所有维度的损失求和
        # total_loss = torch.sum(losses)
        log_var_rosetta = self.log_var_rosetta.to(output.device)
        # 分别计算每个维度的MSE损失
        # self.csv_row = []
        mse_losses = (output - target) ** 2
        
        # 引入不确定性参数
        loss = (mse_losses/(2 * log_var_rosetta.exp())+log_var_rosetta.exp()/2).sum()
        # loss.backward()
        # self.csv_row.extend(mse_losses)
        # self.csv_row.extend(mse_losses/(2 * log_var_rosetta.exp())+log_var_rosetta.exp()/2)
        # print(f'mse_losses:{mse_losses},loss:{loss},loss_func:{mse_losses/(2 * log_var_rosetta.exp())+log_var_rosetta.exp()/2},model_output:{output},target:{target}')
        
        self.csv_row.append(f'mse_losses:{mse_losses},loss:{loss},loss_func:{mse_losses/(2 * log_var_rosetta.exp())+log_var_rosetta.exp()/2},model_output:{output},target:{target}')
        self.csv_writer.writerow(self.csv_row)
        self.csv_row = []
        return loss

    def total_loss(self, model_output, targets):
        """
        根据 train_mode 计算总损失
        """
        cut_outputs = self.cut(model_output)
        loss_Rosetta = 0
        loss_TF = 0
        loss_kd = 0
        loss_IC50 = 0
        loss_content = []
        targets = self.cut(targets)
        
        for i in self.train_target:
            if i == 'RosettaScore':
                loss_Rosetta = self.MSE_loss(cut_outputs[0], targets[0])
                loss_content.append(loss_Rosetta)
                del loss_Rosetta
                continue
            if i == 'TF':
                loss_TF = self.BCE_loss(cut_outputs[1], targets[1])
                loss_content.append(loss_TF)
                del loss_TF
                continue
            if i == 'kd':
                loss_kd = self.MSE_loss(cut_outputs[2], targets[2])
                loss_content.append(loss_kd)
                del loss_kd
                continue
            if i == 'IC50':
                loss_IC50 = self.BCE_loss(cut_outputs[3], targets[3])
                loss_content.append(loss_IC50)
                del loss_IC50
                continue
            if i not in ['RosettaScore','TF','kd','IC50']:
                raise ValueError("train_target must be one of ['RosettaScore','TF','kd','IC50']")
        
        return loss_content

def load_model_from_checkpoint(checkpoint_path=None,
                               device = None 
                               ):
    model = Regression_Module()
    if checkpoint_path:
         # 使用 safetensors.torch 加载模型权重
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
    return model
        
