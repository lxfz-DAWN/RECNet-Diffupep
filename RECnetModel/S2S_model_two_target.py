# %%
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

# accelerator = Accelerator(mixed_precision="fp16")
# device = accelerator.device
device = ('cuda:1')

'''
继承自 2.0的baseline, 缩小了模型的规模，增加了对TF数据训练的支持
'''
# /home/users/hcdai/miniconda3/envs/ESMC/bin/accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 /home/users/hcdai/AI-peptide/Seq2Score/Seq2Rosscore/ESMC-MLP/ESMC-attn/ESMC_attn_2.0_baseline.py

output_ESMC = pd.read_csv("/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/dataset/dataset/ESMC_Embedding_512/generated_EK_withpath_train_512.csv")
score_columns = [
        'scores_total_score']
checkpoint_path = None # 如果没有预训练模型文件，则设置为 None
output_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/checkpoint/20250219-21.58/model/512'
logging_dir  = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/log'

target = ['Ros']

# %%
def ESMC_func(sequence:str):
   '''调用ESMC模型进行embedding
   
   输入：蛋白质序列
   输出：embedding后的隐层特征（Size = [36, 1, 68, 1157]）
   '''
   protein = ESMProtein(sequence= sequence)
   # client = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"
   client = ESMC.from_pretrained("esmc_600m").to('cuda') # or "cpu"
   protein_tensor = client.encode(protein)
   logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
   return logits_output.hidden_states.to('cpu')

# %% [markdown]
# # 负责数据的读取和处理

# %%
def ESMC_sequence_dealing(seq:str,
                          target_seq:str,
                          Receptor_as_Trimers:bool=True
                          ):
    result = ''
    if Receptor_as_Trimers:
        result = seq + '|' + target_seq
    else:
        result = seq + '|' + target_seq
    return result


# %%
score_columns = [
        'scores_total_score', 'scores_dslf_fa13',
        'scores_fa_atr', 'scores_fa_dun', 'scores_fa_elec',
        'scores_fa_intra_rep', 'scores_fa_intra_sol_xover4', 'scores_fa_rep',
        'scores_fa_sol', 'scores_hbond_bb_sc', 'scores_hbond_lr_bb',
        'scores_hbond_sc', 'scores_hbond_sr_bb', 'scores_linear_chainbreak',
        'scores_lk_ball_wtd', 'scores_omega', 'scores_overlap_chainbreak',
        'scores_p_aa_pp', 'scores_pro_close', 'scores_rama_prepro',
        'scores_ref', 'scores_yhh_planarity', 
        'pack_total_score', 'pack_complex_normalized', 'pack_dG_cross',
        'pack_dG_cross/dSASAx100', 'pack_dG_separated',
        'pack_dG_separated/dSASAx100', 'pack_dSASA_hphobic',
        'pack_dSASA_int', 'pack_dSASA_polar', 'pack_delta_unsatHbonds',
        'pack_dslf_fa13', 'pack_fa_atr', 'pack_fa_dun', 'pack_fa_elec',
        'pack_fa_intra_rep', 'pack_fa_intra_sol_xover4', 'pack_fa_rep',
        'pack_fa_sol', 'pack_hbond_E_fraction', 'pack_hbond_bb_sc',
        'pack_hbond_lr_bb', 'pack_hbond_sc', 'pack_hbond_sr_bb',
        'pack_hbonds_int', 'pack_lk_ball_wtd', 'pack_nres_all',
        'pack_nres_int', 'pack_omega', 'pack_p_aa_pp', 'pack_packstat',
        'pack_per_residue_energy_int', 'pack_pro_close', 'pack_rama_prepro',
        'pack_ref', 'pack_sc_value', 'pack_side1_normalized',
        'pack_side1_score', 'pack_side2_normalized', 'pack_side2_score',
        'pack_yhh_planarity']

class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df, 
                 score_columns, 
                 esmc_hidden_column,
                 TF,
                 target = ["Ros","TF"]):
        """
        初始化数据集类。
        :param df: 包含数据的 Pandas DataFrame
        :param score_columns: 包含所有 score 列名的列表
        :param esmc_hidden_column: 包含 ESMC 隐藏层特征文件路径的列名
        """
        self.df = df
        self.score_columns = score_columns
        self.esmc_hidden_column = esmc_hidden_column
        self.TF = TF
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 数据索引
        :return: 包含输入特征和回归目标的字典
        """
        # 读取所有 score 列的数据
        scores = self.df.loc[idx, self.score_columns].values
        scores = scores.astype(float)
        scores = torch.tensor(scores, dtype=torch.bfloat16)

        # 读取 ESMC 隐藏层特征文件路径
        path = self.df.loc[idx, self.esmc_hidden_column]
        ESMC_Hidden = torch.load(path, weights_only=True).to('cpu')
        ESMC_Hidden = ESMC_Hidden[-1, :, :, :]
        
         # 读取TF对应的label
        TF = self.df.loc[idx, self.TF]
        if isinstance(TF, (int, float, np.float64)):
            TF = np.array([TF]).astype(float)
        else:
            TF = TF.values.astype(float)
        TF = torch.tensor(TF, dtype=torch.bfloat16)

        data = {'x': ESMC_Hidden, "labels": scores , "TF":TF, 'target': self.target}
        return data





# %%

# %%
class AttentionRegressionNet(nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, num_blocks=3):
        super(AttentionRegressionNet, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.Linear = nn.Linear(1152, 1024)

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=input_dim * 4,
                                                   activation = "relu",
                                                   batch_first = True,
                                                   norm_first = True)
        # 定义 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, input_dim))
        self.fc1 = nn.Sequential(   
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.005),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.005),
            nn.Linear(4096, 62)     
        )
        self.fc2 = nn.Sequential(   
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.005),
            nn.Linear(4096, 4096),
            nn.Dropout(p = 0.005),
            nn.Linear(4096, 2)     
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, labels=None, TF = None, target = ['Ros','TF']):
        x = self.Linear(x)
        # 调整输入形状以适应 Transformer 编码器的输入要求 [seq_len, batch_size, input_dim]
        batch_size, _, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # 调整形状回原来的 [batch_size, 1, seq_len, input_dim]
        x = x.unsqueeze(1)

        x = self.global_avg_pool(x).squeeze(1).squeeze(1).squeeze(1)
        
        
        x_1 = self.fc1(x)
        x_2 = self.fc2(x)
        x_2 = self.softmax(x_2)
        
        loss = []
        if target:
            current_target = target[0]
            for i in current_target:
                if i == 'Ros':
                    loss_fn = torch.nn.MSELoss()
                    loss.append(loss_fn(x_1, labels))
                    continue
                if i == 'TF':
                    if x_2.shape[1] == 2:
                    # 这里假设 target 为 0 或 1，取对应位置的概率
                        probabilities = x_2.gather(1, TF.long().unsqueeze(1)).squeeze(1)
                    # 计算 BCE 损失
                        loss.append(torch.nn.BCEWithLogitsLoss(probabilities, TF.squeeze(1)))
                    else:
                        raise ValueError("The number of classes in the model output is not equal to 2.")
        
        return loss, x_1, x_2
                
        # if labels is not None:
        #     loss_fn = torch.nn.MSELoss()
        #     loss = loss_fn(x, labels)
        #     return loss, x
        # else:
        #     return x

# %%

def load_model_from_checkpoint(checkpoint_path=None):
    model = AttentionRegressionNet()
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 移除 'module.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 'module.'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    return model

def custom_collate_fn(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key == 'target':
            new_batch[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            new_batch[key] = default_collate([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float, np.float64)):
            values = [item[key] for item in batch]
            if any(np.isnan(val) for val in values):
                values = [0 if np.isnan(val) else val for val in values]
            new_batch[key] = torch.tensor(values)
        else:
            new_batch[key] = [item[key] for item in batch]
    return new_batch
# %%

model = load_model_from_checkpoint(checkpoint_path)
model.to(device).to(torch.bfloat16) 

# %% [markdown]
# # 训练参数

# %%
training_args = TrainingArguments(
    output_dir= output_dir,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=1500,
    weight_decay=0.01,
    logging_dir= logging_dir,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
     # 启用混合精度训练
)

# %% [markdown]
# # 自定义Trainer以处理返回的损失和输出

# %%
# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch = None):
#         # print("Inputs:", inputs['labels'].size)
#         labels = inputs.pop("labels").to(device)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         outputs = model(**inputs, labels=labels)
#         loss = outputs[0]
        
#         # 梯度裁剪
#         clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         return (loss, outputs[1]) if return_outputs else loss

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

    def training_step(self, model, inputs, num_items_in_batch):
        loss = super().training_step(model, inputs, num_items_in_batch)
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        return loss
        # 梯度裁剪
        
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch = None):
        # print("Inputs:", inputs['labels'].size)
        labels = inputs.pop("labels").to(device)
        TF = inputs.pop("TF").to(device)
        target = inputs.pop("target")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=labels,TF = TF, target = target)
        loss = torch.sum(torch.stack(outputs[0]))
        # print(f'label:{labels}output{outputs}loss:{loss}')
        
        return (loss, outputs[1]) if return_outputs else loss

    def on_epoch_end(self):
        # 获取当前 epoch 的验证损失
        eval_metrics = self.evaluate()
        val_loss = eval_metrics["eval_loss"]
        # 根据验证损失调整学习率
        self.lr_scheduler.step(val_loss)

    

# %%

train_df, valid_df = train_test_split(output_ESMC, test_size=0.1, random_state=42)
    
    # 重置索引，保证索引连续，避免出现KeyError
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

scaler = StandardScaler()
train_df[score_columns] = scaler.fit_transform(train_df[score_columns])
valid_df[score_columns] = scaler.transform(valid_df[score_columns])

train_dataset = PeptideDataset(df = train_df,
                                   score_columns = score_columns,
                                   esmc_hidden_column = 'ESMC_Embedding_Path',
                                   TF = 'TF',
                                   target = target)

valid_dataset = PeptideDataset(df = valid_df,
                                   score_columns = score_columns,
                                   esmc_hidden_column = 'ESMC_Embedding_Path',
                                   TF = 'TF',
                                   target = target)

trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=custom_collate_fn,
    )

    # 数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size,
                                                   shuffle=True, pin_memory=True)
eval_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_args.per_device_eval_batch_size,
                                                  shuffle=False, pin_memory=True)

    # 使用加速器包装模型和数据加载器
# model = accelerator.prepare(model)
# train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

# 训练循环
for epoch in range(training_args.num_train_epochs):
  

    # 开始训练
    trainer.train()

    # 保存模型
    model_save_path = f'/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/checkpoint/20250219-21.58/model/512/attn_baseline_checkpoint_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_save_path)

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Epoch {epoch + 1} Evaluation results: {eval_results}")
