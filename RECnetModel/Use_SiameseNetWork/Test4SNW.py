#!/home/users/hcdai/miniconda3/envs/ESMC/bin/python
# -*- coding: utf-8 -*-

# %%
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
from torch.autograd import Variable   
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from accelerate import Accelerator



print("base import done")

# %%
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
print("esm import done")

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
print("device:",torch.cuda.is_available())


# %%
# # git上的示例代码，可以调用ESMC模型进行embedding
# protein = ESMProtein(sequence="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC")
# client = ESMC.from_pretrained("esmc_600m").to("cpu") # or "cpu"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(
#    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True,return_hidden_states=True)
# )
# print(logits_output.hidden_states.shape)
# # 这个hidden_states就是输出的隐层特征

def ESMC_embedding(sequence:str):
   '''调用ESMC模型进行embedding
   
   输入：蛋白质序列
   输出：embedding后的隐层特征（Size = [36, 1, 68, 1152]）
   '''
   protein = ESMProtein(sequence=sequence)
   client = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"
   protein_tensor = client.encode(protein)
   logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
   return logits_output.hidden_states

def show_plot(iteration,loss,label):
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(iteration, loss, color='blue', linewidth=2)  # 绘制损失曲线
    plt.title(label, fontsize=16)  # 设置标题
    plt.legend()  # 添加图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自适应布局
    plt.show()  # 显示图表


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask = None):
        '''接受四个参数：queries、keys、values 和一个可选的 mask。'''

        # 将 queries 张量的形状解包为 b（批量大小）、n（序列长度）和 _（嵌入维度），并将头数分配给 h。
        b, n, _, h = *queries.shape, self.heads


        # 对相应的输入应用 query、key 和 value 线性变换。
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)


        # 将变换后的查询、键和值重新塑造为维度为 (batch_size, num_heads, sequence_length, dim_per_head)。
        # 在 view 方法中的 -1 表示该维度将自动计算。
        # transpose 方法用于交换 num_heads 和 sequence_length 维度。
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)

        # 计算查询和键的缩放点积。
        # einsum 函数在 queries 的最后维度和 keys 的倒数第二维度之间执行批量矩阵乘法。
        # 结果是一个形状为 (batch_size, num_heads, sequence_length, sequence_length) 的张量。
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale

        # 如果提供了掩码，它会被应用到缩放点积张量上。
        # 首先将掩码扁平化，然后在左侧用 True 值填充。然后我们检查掩码是否有正确的维度。
        # 掩码扩展到与 dots 张量具有相同的维度，并用于将 dots 中对应的值替换为 -inf，其中掩码为 False。
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))


        # 对 dots 张量沿着最后的维度应用 softmax 函数。产生注意力权重。
        attn = dots.softmax(dim=-1)

        # 将输出计算为值的加权和，其中权重由注意力分数给出。
        out = torch.einsum('bhij,bhjd->bhid', attn, values)

        # 最后，我们将输出张量重新塑造回输入嵌入的原始形状并返回。
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out

class OnlyCrossAttention(nn.Module):
    def __init__(self, 
                 embedding_dim = 1152, 
                 NormalizedSequenceLength = 256, 
                #  cross_dim = 512,
                 heads = 8):        
        super(OnlyCrossAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.NormalizedSequenceLength = NormalizedSequenceLength
        # self.cross_attention = CrossAttention(cross_dim, heads)
        self.cross_attention = CrossAttention(embedding_dim, heads)

        # self.linear_layers1 = nn.Sequential(
        #     nn.Linear(embedding_dim, cross_dim),
        #     nn.ReLU()
        #     )

    def _ligand_forward(self, x:torch.Tensor):
        '''
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ESMC_size, 1, sequence_length, embedding_dim]。

        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, sequence_length, embedding_dim]
        '''
        x = x.squeeze(2)  # 去掉第二维度 [batch_size, ESMC_size, sequence_length, embedding_dim]
        x = x[:, 0, :, :]  # 取 ESMC_size 的第一层，形状为 [batch_size, sequence_length, embedding_dim]
        # x = x.linear_layers1(x)  # 线性层 [batch_size, sequence_length, cross_dim)]

        return x

    def _receptor_forward(self, x:torch.Tensor):
        '''
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ESMC_size, 1, sequence_length, embedding_dim]。

        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, sequence_length, embedding_dim]
        '''
        x = x.squeeze(2)  # 去掉第二维度 [batch_size, ESMC_size, sequence_length, embedding_dim]
        x = x[:, 0, :, :]  # 取 ESMC_size 的第一层，形状为 [batch_size, sequence_length, embedding_dim]
        # x = x.linear_layers1(x)  # 线性层 [batch_size, sequence_length, cross_dim)]

        return x



    def forward(self, ligand:torch.Tensor, receptor:torch.Tensor):
        ligand = self._ligand_forward(ligand)
        receptor = self._receptor_forward(receptor)
        out = self.cross_attention(queries = ligand, 
                                   keys = receptor,
                                   values = receptor)
        
        return out  

class OnlyCrossAttentionClassifier(nn.Module):
    '''单独使用CrossAttention的分类器'''
    def __init__(self, embedding_dim=1152, heads=8, NormalizedSequenceLength = 256):
        super(OnlyCrossAttentionClassifier, self).__init__()
        self.only_cross_attention = OnlyCrossAttention(embedding_dim, heads=heads)
        self.linear_layers_classifier = nn.Sequential(
            nn.Linear(embedding_dim * NormalizedSequenceLength, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)  # 假设我们要进行序列分类，输出维度为 1
        )

    def forward(self, ligand:torch.Tensor, receptor:torch.Tensor):
        out:torch.Tensor = self.only_cross_attention(ligand, receptor) # [batch_size, sequence_length, cross_dim)]
        out = out.reshape(out.shape[0], -1)  # [batch_size, sequence_length * cross_dim]
        out = self.linear_layers_classifier(out)
        return out

# %%
class SiameseNetwork(nn.Module):
    '''
    使用 Transformer 架构和多头注意力机制进行序列特征提取，输出特征提取后的信息。
    '''

    def __init__(self, embedding_dim = 1152, NormalizedSequenceLength = 256,):
        """
        初始化 Transformer 架构。

        参数:
            embedding_dim (int): 输入嵌入的维度（默认 1152）。
        """
        super(SiameseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.NormalizedSequenceLength = NormalizedSequenceLength


        self.transformer_encoder_layer1 = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,  # 输入特征维度
            nhead=8,      # 多头注意力头数
            dim_feedforward=512,  # 前馈网络隐藏层维度
            dropout=0.1,  # 丢弃率
            activation='relu'  # 激活函数
        )

        self.transformer_encoder1 = nn.TransformerEncoder(
            self.transformer_encoder_layer1,  # 编码层
            num_layers=2  # 编码层数
        )

        self.linear_layers1 = nn.Sequential(
            nn.Linear(self.NormalizedSequenceLength * 36, 512),
            nn.ReLU()
            )
        

        self.linear_layers2 = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU()
            )
    
        self.transformer_encoder_layer2 = nn.TransformerEncoderLayer(
        d_model = 512,  # 输入特征维度
        nhead=8,      # 多头注意力头数
        dim_feedforward=512,  # 前馈网络隐藏层维度
        dropout=0.1,  # 丢弃率
        activation='relu'  # 激活函数
        )

        self.transformer_encoder2 = nn.TransformerEncoder(
            self.transformer_encoder_layer1,  # 编码层
            num_layers=2  # 编码层数
        )

        self.linear_layers3 = nn.Sequential(
            nn.Linear(512*512, 1024),
            nn.ReLU()
            )




    def _forward_ligand(self, x:torch.Tensor):
        '''
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ESMC_size, 1, sequence_length, embedding_dim]。

        返回:
            torch.Tensor: 输出张量，形状为 
        '''
        x = x.squeeze(2)  # 去掉第二维度 [batch_size, ESMC_size, sequence_length, embedding_dim]
        x = x.reshape(x.size(0), -1, x.size(-1))  # 展平 [batch_size, ESMC_size * sequence_length, embedding_dim]
        x = x.permute(1, 0, 2)  # 转置 [ESMC_size * sequence_length, batch_size, embedding_dim]
        x = self.transformer_encoder1(x)  # 编码器  [ESMC_size * sequence_length, batch_size, embedding_dim] [1800, 32, 1152]
        x = x.permute(1, 2, 0)  # 转置 [batch_size, embedding_dim, ESMC_size * sequence_length]
        x = self.linear_layers1(x)  # 全连接层 [batch_size, embedding_dim , 512]
        x = x.permute(2, 0, 1)  # [512, batch_size, embedding_dim]
        x = self.transformer_encoder1(x)  # 编码器  [512, batch_size, embedding_dim]
        x = self.linear_layers2(x) # 全连接层 [512, batch_size, 512]
        x = self.transformer_encoder_layer2(x)  # 编码器  [512, batch_size, 512]
        x = x.permute(1, 0, 2)   # [batch_size, 512, 512]
        x = x.reshape(x.shape[0], -1)  # [batch_size, 512*512]
        x = self.linear_layers3(x)  # [batch_size, 1024]


        


        return x
    

    def _forward_receptor(self, x:torch.Tensor):
        '''
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ESMC_size, 1, sequence_length, embedding_dim]。

        返回:            
            torch.Tensor: 输出张量，形状为
        '''
        x = x.squeeze(2)  # 去掉第二维度 [batch_size, ESMC_size, sequence_length, embedding_dim]
        x = x.reshape(x.size(0), -1, x.size(-1))  # 展平 [batch_size, ESMC_size * sequence_length, embedding_dim]
        x = x.permute(1, 0, 2)  # 转置 [ESMC_size * sequence_length, batch_size, embedding_dim]
        x = self.transformer_encoder1(x)  # 编码器  [ESMC_size * sequence_length, batch_size, embedding_dim] [1800, 32, 1152]
        x = x.permute(1, 2, 0)  # 转置 [batch_size, embedding_dim, ESMC_size * sequence_length]
        x = self.linear_layers1(x)  # 全连接层 [batch_size, embedding_dim , 512]
        x = x.permute(2, 0, 1)  # [512, batch_size, embedding_dim]
        x = self.transformer_encoder1(x)  # 编码器  [512, batch_size, embedding_dim]
        x = self.linear_layers2(x) # 全连接层 [512, batch_size, 512]
        x = self.transformer_encoder_layer2(x)  # 编码器  [512, batch_size, 512]
        x = x.permute(1, 0, 2)   # [batch_size, 512, 512]
        x = x.reshape(x.shape[0], -1)  # [batch_size, 512*512]
        x = self.linear_layers3(x)  # [batch_size, 1024]

        return x


    def forward(self, input1, input2):
        output1 = self._forward_ligand(input1)
        output2 = self._forward_receptor(input2)
        return output1, output2

class SiameseClassifier(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SiameseClassifier, self).__init__()
        self.siamese_network = SiameseNetwork(embedding_dim=embedding_dim)
        self.distance_metric = nn.PairwiseDistance(p=2)  # 使用欧氏距离
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0或1的概率
        )

    def forward(self, input1, input2):
        output1, output2 = self.siamese_network(input1, input2)
        distance = self.distance_metric(output1, output2).unsqueeze(1)  # 计算距离并调整形状
        return self.fc(distance)  # 通过全连接层进行分类


# %%
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, distances = 'cosine_distance'):
        # label指示两个样本是否相似（1表示不相似，0表示相似）
        if distances == 'euclidean_distance':
            # 欧几里得距离：适合低维稠密数据，捕捉绝对差异
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        elif distances == 'cosine_distance':
            # 余弦距离：适合高维稀疏数据，方向敏感，适合高维稀疏矩阵
            cosine_similarity = F.cosine_similarity(output1, output2, dim=1, eps=1e-6)
            cosine_distance = 1 - cosine_similarity
            loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        elif distances == 'exponential_distance':
            # 指数距离：适合需要强调微小差异的任务
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            exponential_distance = torch.exp(euclidean_distance) - 1
            loss_contrastive = torch.mean((1-label) * torch.pow(exponential_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - exponential_distance, min=0.0), 2))
            # 如果两个输入不相似的程度大于self.margin，即足够不相似，则loss为0，否则loss为平方距离
        else:
            raise ValueError('Unsupported distance metric: {}'.format(distances))



        return loss_contrastive

# %%
# 读取数据
class ProteinPairsDataset(Dataset):
    
    def __init__(self, data, NormalizedSequenceLength, Seq2Tensor_path = './Seq2Tensor', transform=None):
        """
        初始化数据集。

        参数:
            data (DataFrame): 包含序列对及其标签的 DataFrame。
            Seq2Tensor_path (str): 存储张量数据的路径。
        """
        self.data = data  
        self.NormalizedSequenceLength = NormalizedSequenceLength
        self.Seq2Tensor_path = Seq2Tensor_path
        self.ProteinPairsDataTensors = set([path.split('/')[-1].rsplit('.', 1)[0] for path in glob.glob(f"{Seq2Tensor_path}/*pt")]) # 加载已经保存好的Emdeding数据
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def seq2tensor(self, seq):
        # if seq in self.ProteinPairsDataTensors:
        #     return torch.load(f"{self.Seq2Tensor_path}/{seq}.pt")
        # else:
        #     seq_tensor = ESMC_embedding(seq)
        #     try:
        #         torch.save(seq_tensor, f"{self.Seq2Tensor_path}/{seq}.pt")
        #     except:
        #         print(f"Error: {seq} cannot be saved.")
        #     self.ProteinPairsDataTensors.add(seq)
        #     return seq_tensor
        return ESMC_embedding(seq).to(device)
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        获取特定索引的样本。

        参数:
            idx (int): 索引值

        返回:
            dict: 包含序列对及其标签的字典
        """
        sequence1 = self.data.iloc[idx]['sequence1']  # 根据列名获取序列1
        sequence2 = self.data.iloc[idx]['sequence2']  # 根据列名获取序列2
        label = self.data.iloc[idx]['label']  # 根据列名获取标签

        # 将序列转成张量，使用seq2tensor
        sequence1_tensor = self.seq2tensor(sequence1)
        sequence2_tensor = self.seq2tensor(sequence2)

        if self.transform is not None:
            sequence1_tensor = self.transform(sequence1_tensor, self.NormalizedSequenceLength)
            sequence2_tensor = self.transform(sequence2_tensor, self.NormalizedSequenceLength)        

        return {"sequence1": sequence1_tensor, "sequence2": sequence2_tensor, "label": label}

def transform(sequence_tensor, NormalizedSequenceLength = 50):
    """
    判断序列张量是否合规，否则进行变换
    """
    # 输入的形状为[ESM_size, 1, sequence_length, embedding_dim]
    # 输出的形状为[ESM_size, 1, NormalizedSequenceLength, embedding_dim]
    if sequence_tensor.shape[2] > NormalizedSequenceLength:
        sequence_tensor = sequence_tensor[:, :, :NormalizedSequenceLength, :]
    elif sequence_tensor.shape[2] < NormalizedSequenceLength:
        padding_tensor = torch.zeros(sequence_tensor.shape[0], sequence_tensor.shape[1], NormalizedSequenceLength - sequence_tensor.shape[2], sequence_tensor.shape[3])
        sequence_tensor = torch.cat((sequence_tensor, padding_tensor), dim=2)
    return sequence_tensor



# 加载数据
def load_data(ProteinPairsData_csv_path, Seq2Tensor_path, NormalizedSequenceLength =50, batch_size=32, val_size=0.1, test_size=0.1):
    """
    加载数据并划分为训练集、验证集和测试集。

    参数:
        ProteinPairsData_csv_path (str): CSV 文件路径。
        batch_size (int): 每个批次的大小。
        val_size (float): 验证集的比例（0-1之间的浮点数）。
        test_size (float): 测试集的比例（0-1之间的浮点数）。

    返回:
        tuple: 训练集, 验证集和测试集的 DataLoader。
    """
    # 读取 CSV 文件
    data = pd.read_csv(ProteinPairsData_csv_path)

    # 划分训练集和临时集（临时集中将会包含验证集和测试集）
    train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=42)

    # 计算临时集中验证集和测试集的比例
    temp_val_size = val_size / (val_size + test_size)
    
    # 划分验证集和测试集
    val_data, test_data = train_test_split(temp_data, test_size=temp_val_size, random_state=42)

    # 创建数据集实例
    train_dataset = ProteinPairsDataset(data = train_data,
                                        NormalizedSequenceLength = NormalizedSequenceLength,
                                        Seq2Tensor_path = Seq2Tensor_path,
                                        transform = transform)
    val_dataset = ProteinPairsDataset(data = val_data,
                                      NormalizedSequenceLength = NormalizedSequenceLength,
                                      Seq2Tensor_path = Seq2Tensor_path,
                                      transform = transform)
    test_dataset = ProteinPairsDataset(data = test_data,
                                       NormalizedSequenceLength = NormalizedSequenceLength,
                                       Seq2Tensor_path = Seq2Tensor_path,
                                       transform = transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# %%
def save_model(model, optimizer, epoch, file_path):
    """
    保存模型及其状态

    参数:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前训练的epoch
        file_path: 保存的文件路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)

def load_model(model, optimizer, file_path):
    """
    加载模型及其状态

    参数:
        model: 要加载的模型
        optimizer: 优化器
        file_path: 加载的文件路径
    """
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']  # 返回加载的epoch

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
    # criterion = ContrastiveLoss(margin=margin)  # 对比损失
    criterion = nn.BCEWithLogitsLoss()
    

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
            # output1, output2 = model(sequence1, sequence2)
            out = model(sequence1, sequence2)

            # 计算损失
            # loss = criterion(output1, output2, labels)
            loss = criterion(out, target = labels.unsqueeze(1).float())
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
            # output1, output2 = model(sequence1, sequence2)
            out = model(sequence1, sequence2)

            # 计算损失
            # loss = criterion(output1, output2, labels)
            loss = criterion(out, target = labels.unsqueeze(1).float())
            val_loss += loss.item()

    return val_loss / len(val_loader)


# %%
# def test_model(model, test_loader):
#     """
#     测试模型并计算测试损失和准确率。

#     参数:
#         model: 要测试的模型
#         test_loader: 用于测试的数据加载器
#         criterion: 损失函数

#     返回:
#         float: 测试损失
#         float: 测试准确率
#     """
#     model.eval()  # 设置模型为评估模式
#     criterion = ContrastiveLoss()
#     test_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():  # 评估时不需要反向传播
#         for batch in test_loader:
#             sequence1 = batch["sequence1"]
#             sequence2 = batch["sequence2"]
#             labels = batch["label"]

#             # 前向传播
#             output1, output2 = model(sequence1, sequence2)
            
#             # 计算损失
#             loss = criterion(output1, output2, labels)
#             test_loss += loss.item()

#             # 计算准确率，这里假设通过某种方式获取每个样本的预测结果
#             # 需要定义一个方法计算预测结果，比如使用阈值来决定正负样本
#             predicted = (torch.nn.functional.cosine_similarity(output1, output2) > 0.5).float()  # 示例
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     avg_test_loss = test_loss / len(test_loader)
#     accuracy = correct / total * 100  # 转换为百分比

#     print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')
#     return avg_test_loss, accuracy

# %%
   
# ProteinPairsData_csv_path = "/home/users/hcdai/AI-peptide/Seq2Score/SiameseNetWork/est.csv",
# train_batch_size = 64,
# train_number_epochs = 100,
# learning_rate = 0.001, # learning_rate (float): SiameseNetwork的学习率
# distances = "cosine_distance", # distances (str): 距离函数，可选["cosine_distance", "euclidean_distance", "exponential_distance"]
# margin = 2.0 # margin (float): 正负样本的margin值，用于loss计算
# val_size = 0.1 # val_size (float): 验证集的比例（0-1之间的浮点数）
# test_size = 0.1 # test_size (float): 测试集的比例（0-1之间的浮点数）

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
# model = SiameseNetwork()
model = OnlyCrossAttentionClassifier()
# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 指定保存路径
model_save_path = 'model_RS_PDB_CA1.pth'

# 加载已保存的模型
# load_model(model, optimizer, model_save_path)

# 训练模型
train_model(model = model, 
            train_loader = train_loader, 
            val_loader = val_loader, 
            num_epochs = SNW_Config["train_number_epochs"], 
            learning_rate = SNW_Config['learning_rate'], 
            margin = SNW_Config['margin'],
            model_save_path = model_save_path)

# # 测试模型
# try:
#     load_model(model, optimizer, model_load_path = model_save_path)
# except FileNotFoundError:
#     print('没有找到模型，要进行测试时，请确保模型文件存在。')

# # 测试模型
# test_model(model, test_loader)




