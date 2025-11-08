from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import torch.nn as nn
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel, Trainer, TrainingArguments
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from esm.tokenization import get_esmc_model_tokenizers
import os
'''
初始化accelerator，配置device变量
'''
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
# device = ('cuda:1')
# 用ESM2试试？
# /home/users/hcdai/miniconda3/envs/ESMC/bin/accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 /home/users/hcdai/AI-peptide/Seq2Score/Henya/model/attn_baseline/Attn_baseline.py
# CUDA_VISIBLE_DEVICES="1" accelerate launch --mixed_precision=fp16 --num_processes=1 /home/users/hcdai/AI-peptide/Seq2Score/Henya/model/attn_baseline/Attn_baseline.py
'''
参数
'''
ESMC_path = "/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250302_baseline/ESMC_path/esmc_model.pth"
acc_or_not = True
ESMC_train = True
model_lr = 1e-5
esmc_model_lr = 1e-5
model_path = None
output_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250308_test_finetuning/output_dir'
logging_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250308_test_finetuning/output_dir'
input_df = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/dataset/panpep_test/finetuning_test/for_train.csv'
model_save_path = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250308_test_finetuning/model_path/fintuned_baseline1.pth'
ESMC_save_path = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250308_test_finetuning/ESMC_path/fintuned_baseline1_esmc.pth'
input_df = pd.read_csv(input_df)


# 确保目录存在（原代码正确）
model_dir = os.path.dirname(model_save_path)
ESMC_dir = os.path.dirname(ESMC_save_path)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(ESMC_dir, exist_ok=True)

def ESMC_600M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True, model_path: str = None):
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
ESMC_model = ESMC_600M_202412(device = device, use_flash_attn= True,model_path = ESMC_path)
if ESMC_train == True:
    ESMC_model.train()


def ESMC_encoding(sequence:str):
    '''
    调用ESMC进行Encoding，但是不进行计算，用途是输出forward方法可以接受的张量
    '''
    protein = ESMProtein(sequence = sequence)
    if acc_or_not == True:
        protein_tensor = ESMC_model.module.encode(protein)
    else:
        protein_tensor = ESMC_model.encode(protein)
    return protein_tensor


class PeptideDataset(torch.utils.data.Dataset):
    """
    定义数据集类，加载peptide和receptor的sequence数据，输出label和Peptide对应的ESMProtein对象
    """
    def __init__(self,
                 df,
                 peptide, 
                 receptor,
                 label):
        self.df = df
        self.peptide = peptide
        self.receptor = receptor
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, 
                    idx):
        peptide = self.df.loc[idx, self.peptide]
        receptor = self.df.loc[idx, self.receptor]
        merge = receptor + "|" + peptide
        merge = ESMC_encoding(merge)
        
        label = self.df.loc[idx, self.label]
        if label == 1:
            label = torch.tensor([1,0], dtype=torch.float)
        else :
            label = torch.tensor([0,1], dtype=torch.float)

        data = {'merge': merge , "label": label}
        return data

# class ESMC_embedding(nn.Module):
#     def __init__(self,
#                  ESMC=ESMC_model):
#         super(ESMC_embedding,self).__init__()
#         self.ESMC_model = ESMC
    
#     def forward(self,merge):
#         merge = self.ESMC_model.logits(merge,LogitsConfig(sequence = True,return_hidden_states = True))
#         merge = merge.hidden_states[-1, :, :, :]
#         return merge
# esmc_embedding = ESMC_embedding(ESMC = ESMC_model).to(device) 

def ESMC_embedding(input_list:list,model = ESMC_model):
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

class Probability_Prediction_Net(nn.Module):
    def __init__(self, 
                 input_dim=2048, 
                 num_heads=8, 
                 num_blocks=12, 
                 ESMC_d_model = 1152,
                 ESMC = ESMC_model):
        super(Probability_Prediction_Net, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.Linear = nn.Sequential(   
            nn.Linear(ESMC_d_model, input_dim*4),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(input_dim*4, input_dim*4),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(input_dim*4, input_dim)     
        )
        # self.ESMC = ESMC

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=input_dim * 4,
                                                   activation = "relu",
                                                   batch_first = True,
                                                   norm_first = True)
        # 定义 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks) 
        self.fc1 = nn.Sequential(   
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4), 
            nn.ReLU(), 
            nn.Dropout(p = 0.4),
            nn.Linear(input_dim*4, input_dim*4),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim*4), 
            nn.Dropout(p = 0.4),
            nn.Linear(input_dim*4, 2)   
        )

    def forward(self, merge, label=None):
        # merge = self.ESMC.logits(merge,LogitsConfig(sequence = True,return_hidden_states = True))
        # merge = merge.hidden_states[-1, :, :, :].shape
        
        x = self.Linear(merge)
        # 调整输入形状以适应 Transformer 编码器的输入要求 [seq_len, batch_size, input_dim]
        batch_size, _, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)
        x = x[ :, 0, :]
        x = self.fc1(x)
        

        if label is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()# 使用二元交叉熵损失函数
            loss = loss_fn(x, label)
            return loss, x
        else:
            return x

def load_model_from_checkpoint(checkpoint_path=None):
    '''
    加载概率模型的函数
    '''
    model = Probability_Prediction_Net()
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
model = load_model_from_checkpoint(model_path)
model.to(device)

training_args = TrainingArguments(
    output_dir= output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir= logging_dir,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True,  # 启用混合精度训练
)

optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': model_lr},
    {'params': ESMC_model.parameters(), 'lr': esmc_model_lr}
], weight_decay=0.01)
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
        label = inputs.pop("label").to(device)
        inputs = {k: v for k, v in inputs.items()}
        # 处理 merge 列表中的每个元素
        
        # merge = [item.to(device) for item in inputs['merge']]
        # 找出最大长度
        # max_length = max([item.sequence.shape[1] for item in merge])

        # # 对每个序列进行填充
        # padded_merge = []
        # for single_merge in merge:
        #     current_length = single_merge.sequence.shape[1]
        #     padding_length = max_length - current_length
        #     padding = torch.ones((single_merge.sequence.shape[0], padding_length, single_merge.sequence.shape[2]), device=device)
        #     padded_single_merge = torch.cat([single_merge, padding], dim=1)
        #     padded_merge.append(padded_single_merge)

        # # 依次处理每个元素
        # all_outputs = []
        # for single_padded_merge in padded_merge:
        #     single_output = esmc_embedding(single_padded_merge)
        #     all_outputs.append(single_output)

        # # 将所有输出合并成一个张量
        # outputs = torch.stack(all_outputs)
        inputs = ESMC_embedding(input_list = inputs['merge'])
        outputs = model(merge = inputs, label=label)
        loss = outputs[0]
        # print(f'label:{label}output{outputs}loss:{loss}')
        
        return (loss, outputs[1]) if return_outputs else loss

    def on_epoch_end(self):
        # 获取当前 epoch 的验证损失
        eval_metrics = self.evaluate()
        val_loss = eval_metrics["eval_loss"]
        # 根据验证损失调整学习率
        self.lr_scheduler.step(val_loss)

# 自定义数据整理函数
def custom_collate_fn(batch):
    merge = [item['merge'] for item in batch]
    labels = [item['label'] for item in batch]
    # 处理 labels 中的 None 值
    labels = [label if label is not None else torch.tensor(float('nan')) for label in labels]
    labels = torch.stack(labels)
    return {'merge': merge, 'label': labels}

train_df, valid_df = train_test_split(input_df, test_size=0.1, random_state=1)
# 重置索引，保证索引连续，避免出现KeyError
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

train_dataset = PeptideDataset(df = train_df,
                               peptide = 'peptide',
                               receptor = 'binding_TCR',
                               label = 'label')

valid_dataset = PeptideDataset(df = valid_df,
                               peptide = 'peptide',
                               receptor = 'binding_TCR',
                               label = 'label')

trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=custom_collate_fn
    )

    # 数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size,
                                                   shuffle=True, pin_memory=True)
eval_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_args.per_device_eval_batch_size,
                                                  shuffle=False, pin_memory=True)

    # 使用加速器包装模型和数据加载器
model = accelerator.prepare(model)
train_dataloader, ESMC_model, eval_dataloader = accelerator.prepare(train_dataloader, ESMC_model, eval_dataloader)

# 训练循环
for epoch in range(training_args.num_train_epochs):

    # 开始训练
    trainer.train()

    # 保存模型
    
    torch.save(model.state_dict(), model_save_path)
    torch.save(ESMC_model.state_dict(),ESMC_save_path)

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Epoch {epoch + 1} Evaluation results: {eval_results}")
    