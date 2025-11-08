#!/home/users/hcdai/miniconda3/envs/ESMC/bin/python
# -*- coding: utf-8 -*-

# ============= 导入所需的包 ==============
import torch
from torch import optim
import json
from tqdm import tqdm
from accelerate import Accelerator

print("base import done")



accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
print("device:",torch.cuda.is_available())

# 导入模型
from model_all import SiameseNetwork, ContrastiveLoss
from dm import *




# %%
# 定义训练循环

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, margin, model_save_path=None):
    """
    训练模型

    参数:
        model: 要训练的模型
        train_loader: 用于训练的数据加载器
        val_loader: 用于验证的数据加载器
        num_epochs: 训练的轮数
        learning_rate: 学习率
        margin: 正负样本的margin值，用于loss计算
    """
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 使用 accelerator.prepare 准备模型和优化器
    model, optimizer = accelerator.prepare(model, optimizer)

    # 定义损失函数类
    criterion = ContrastiveLoss(margin=margin)  # 对比损失
    

    train_losses = []  # 用于记录训练损失
    val_losses = []    # 用于记录验证损失

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0
        
        # for batch in train_loader:
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'): # 使用 tqdm 包装 train_loader，显示进度
            sequence1 = batch["sequence1"].to(device)
            sequence2 = batch["sequence2"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()  # 清零梯度
            
            # 前向传播
            output1, output2 = model(sequence1, sequence2)

            # 计算损失
            loss = criterion(output1, output2, labels)
            # loss.backward()  # 反向传播
            accelerator.backward(loss)
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()

        # 验证
        avg_train_loss = train_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion)

        # 记录损失
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 使用 show_plot 函数进行损失可视化

        
        # 每个epoch后保存模型
        if model_save_path:
            save_model(model, optimizer, epoch, model_save_path)  # 每个epoch后保存模型


    

def validate_model(model, val_loader, criterion):
    """
    验证模型

    参数:
        model: 要验证的模型
        val_loader: 用于验证的数据加载器
        criterion: 损失函数
    """
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0

    with torch.no_grad():  # 评估时不需要反向传播
        for batch in val_loader:
            sequence1 = batch["sequence1"].to(device)
            sequence2 = batch["sequence2"].to(device)
            labels = batch["label"].to(device)

            # 前向传播
            output1, output2 = model(sequence1, sequence2)

            # 计算损失
            loss = criterion(output1, output2, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# def test_model(model, test_loader):
#     """
#     测试模型

#     参数:
#         model: 要测试的模型
#         test_loader: 用于测试的数据加载器
#     """
#     model.eval()  # 设置模型为评估模式
#     test_loss = 0.0

#     with torch.no_grad():  # 评估时不需要反向传播
#         for batch in test_loader:
#             sequence1 = batch["sequence1"].to(device)
#             sequence2 = batch["sequence2"].to(device)
#             labels = batch["label"].to(device)

#             # 前向传播
#             output1, output2 = model(sequence1, sequence2)

#             # 计算损失
#             loss = criterion(output1, output2, labels)
#             test_loss += loss.item()

#     return test_loss / len(test_loader)


# %%
# 加载config文件
config_json_path = '/home/users/hcdai/AI-peptide/Seq2Score/SiameseNetWork/SNW_config.json'
with open(config_json_path, 'r') as f:
    SNW_Config = json.load(f)
print("config successful")


# 加载数据
train_loader, val_loader, test_loader = load_data(ProteinPairsData_csv_path = SNW_Config["ProteinPairsData_csv_path"], 
                                                  Seq2Tensor_path=SNW_Config["Seq2Tensor_path"], 
                                                  NormalizedSequenceLength = SNW_Config["NormalizedSequenceLength"], 
                                                  batch_size=32, 
                                                  val_size=0.1, 
                                                  test_size=0.1)


# 实例化模型
model = SiameseNetwork()
# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 指定保存路径
model_save_path = 'weights/model_RS_PDB_SN1.pth'

# 加载已保存的模型
load_model(model, optimizer, model_save_path)

# 训练模型
train_model(model = model, 
            train_loader = train_loader, 
            val_loader = val_loader, 
            num_epochs = SNW_Config["train_number_epochs"], 
            learning_rate = SNW_Config['learning_rate'], 
            margin = SNW_Config['margin'],
            model_save_path = model_save_path)





