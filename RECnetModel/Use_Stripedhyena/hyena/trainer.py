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
import yaml

from models import HyenaClassification
from stripedhyena.utils import dotdict

'''
初始化accelerator，配置device变量
'''
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
# device = ('cuda:1')
# 用ESM2试试？
# /home/users/hcdai/miniconda3/envs/ESMC/bin/accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 /home/users/hcdai/AI-peptide/Seq2Score/Henya/model/attn_baseline/Attn_baseline.py
# CUDA_VISIBLE_DEVICES="1" accelerate launch --mixed_precision=fp16 --num_processes=1 /home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/trainer.py
'''
参数
'''
ESMC_path = "/home/users/hcdai/AI-peptide/Seq2Score/Henya/weights/20250302_baseline/ESMC_path/esmc_model.pth"
acc_or_not = True
ESMC_train = True
model_lr = 1e-5
esmc_model_lr = 1e-5
model_path = None
output_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/weights/20250323_test_finetuning/output_dir'
logging_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/weights/20250323_test_finetuning/output_dir'
input_df = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/dataset/panpep_test/Majority_from_panpep_test.csv'
model_save_path = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/weights/20250323_test_finetuning/model_path/fintuned_baseline1.pth'
ESMC_save_path = '/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/weights/20250323_test_finetuning/ESMC_path/fintuned_baseline1_esmc.pth'
input_df = pd.read_csv(input_df)

print("参数初始化成功")

# 确保目录存在（原代码正确）
model_dir = os.path.dirname(model_save_path)
ESMC_dir = os.path.dirname(ESMC_save_path)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(ESMC_dir, exist_ok=True)

print("目录创建成功")




class PeptideDataset(torch.utils.data.Dataset):
    """
    定义数据集类，加载peptide和receptor的sequence数据，输出label和Peptide对应的ESMProtein对象
    """
    def __init__(self,
                 df,
                 peptide, 
                 receptor,
                 label,
                 tokenizer):
        self.df = df
        self.peptide = peptide
        self.receptor = receptor
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, 
                    idx):
        peptide = self.df.loc[idx, self.peptide]
        receptor = self.df.loc[idx, self.receptor]
        merge = receptor + "|" + peptide
        # print(merge)
        merge = self.tokenizer.tokenize(merge)
        
        label = self.df.loc[idx, self.label]
        if label == 1:
            label = torch.tensor([1,0], dtype=torch.float32)
        else :
            label = torch.tensor([0,1], dtype=torch.float32)
        
        data = {'merge': merge , "label": label}
        
        # print(data)
        return data





# def load_model_from_checkpoint(checkpoint_path=None):
#     '''
#     加载概率模型的函数
#     '''
#     model = Probability_Prediction_Net()
#     if checkpoint_path:
#         state_dict = torch.load(checkpoint_path, map_location=device)
#         # 移除 'module.' 前缀
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('module.'):
#                 new_key = key[7:]  # 移除 'module.'
#                 new_state_dict[new_key] = value
#             else:
#                 new_state_dict[key] = value
#         model.load_state_dict(new_state_dict)
#     return model
# model = load_model_from_checkpoint(model_path)

# 加载模型

Hconfig_path = "/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/hyena_config.yml"

with open(Hconfig_path, "r") as f:
    Hconfig = dotdict(yaml.safe_load(f), Loader = yaml.FullLoader)

model = HyenaClassification(Hconfig)
tokenizer = model.hyena.tokenizer


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
    # {'params': ESMC_model.parameters(), 'lr': esmc_model_lr}
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
        
        outputs = model(input_ids = inputs, labels = label)
        loss = outputs["loss"]
        # print(f'label:{label}output{outputs}loss:{loss}')
        
        return (loss, outputs["logits"]) if return_outputs else loss

    def on_epoch_end(self):
        # 获取当前 epoch 的验证损失
        eval_metrics = self.evaluate()
        val_loss = eval_metrics["eval_loss"]
        # 根据验证损失调整学习率
        self.lr_scheduler.step(val_loss)

# 自定义数据整理函数
def custom_collate_fn(batch):
    # print(batch)
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
                               label = 'label',
                               tokenizer = tokenizer)

valid_dataset = PeptideDataset(df = valid_df,
                               peptide = 'peptide',
                               receptor = 'binding_TCR',
                               label = 'label',
                               tokenizer = tokenizer)

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

# acc
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

print("开始训练")

# 训练循环
for epoch in range(training_args.num_train_epochs):

    # 开始训练
    # trainer.train()
    for batch in train_dataloader:
        print(batch)
        # trainer.train_step(batch)
        break
    break

    # 保存模型
    
    torch.save(model.state_dict(), model_save_path)
    torch.save(ESMC_model.state_dict(),ESMC_save_path)

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Epoch {epoch + 1} Evaluation results: {eval_results}")
    