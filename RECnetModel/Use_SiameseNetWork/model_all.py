#!/home/users/hcdai/miniconda3/envs/ESMC/bin/python
# -*- coding: utf-8 -*-

import torch   
import torch.nn as nn
import torch.nn.functional as F


# ============== CrossAttention模型 ==============

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

# ============== SiameseNetWork模型 ==============
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

# ============== ContrastiveLoss ==============
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