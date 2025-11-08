'''
Content: 
    Self-Attention Module
    Cross-Attention Module
    Regressiong Modules
    疑似有cuda out of memory问题
'''
import torch
import torch.nn as nn
import pandas as pd
import torch
# from S2S_utils import GradNormModel


class SelfAttention_Module(nn.Module):
    '''
    Self-Attention block
    Dealing with tensor given by ESMC of protein complex
    [batch_size, 1,seq_length, hidden_size] -> [batch_size, seq_length, hidden_size]
    
    args:
    input_dim=2048,
    num_heads=8,
    num_blocks=12
    '''
    def __init__(self, input_dim=2048, num_heads=8, num_blocks=12):
        super(SelfAttention_Module, self).__init__()
        self.Linear = nn.Linear(1152, 2048)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=input_dim * 4,
                                                   activation = "relu",
                                                   batch_first = True,
                                                   norm_first = True)
        # 定义 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, x):
        x = self.Linear(x)
        # 调整输入形状以适应 Transformer 编码器的输入要求 [seq_len, batch_size, input_dim]
        batch_size, _, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)
        
        return x

# 测试SelfAttention_Module。[64,1,256,1152] -> [64, 256, 2048]
# model = SelfAttention_Module().to('cuda')
# batch_size = 64
# seq_len = 256
# input_dim = 1152
# input_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')

# output = model(input_tensor)
# print(output.shape) # [64, 2048]

class CrossAttention_Module(nn.Module):
    '''
    Cross-Attention block+
    Dealing with tensor given by ESMC of ligand and receptor sequences
    Firstly using Cross-Attention block to get the cross-attention weights of ligand and receptor
    Then uses the Self-Attention block to get the final output
    [batch_size, 1,seq_length, hidden_size] -> [batch_size ,seq_length, hidden_size]
    
    args:
    input_dim=2048, 
    num_heads=8,
    num_blocks=6
    '''
    def __init__(self, input_dim=2048, num_heads=8 ,num_blocks=6):
        super(CrossAttention_Module, self).__init__()
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.input_dim = input_dim
        
        self.Linear_ligand_init = nn.Linear(1152, 2048)
        self.Linear_receptor_init = nn.Linear(1152, 2048)
        self.cross_attention_block_0 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_1 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_2 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_3 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_4 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_5 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_6 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.cross_attention_block_7 = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        
        self.Linear_ligand_0 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_ligand_1 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_ligand_2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_ligand_3 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_receptor_0 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_receptor_1 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_receptor_2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )
        self.Linear_receptor_3 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048)
        )

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=input_dim * 4,
                                                   activation = "relu",
                                                   batch_first = True,
                                                   norm_first = True)
        # 定义 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        
    def forward(self, ligand: torch.Tensor, receptor: torch.Tensor):
        ligand = self.Linear_ligand_init(ligand)
        receptor = self.Linear_receptor_init(receptor)
        # 调整输入形状以适应 Transformer 编码器的输入要求 [seq_len, batch_size, input_dim]
        batch_size, _, seq_len, _ = ligand.shape
        ligand = ligand.view(batch_size, seq_len, -1)
        receptor = receptor.view(batch_size, seq_len, -1)
        # 第一层 Cross-Attention
        ligand1, _ = self.cross_attention_block_0(ligand, receptor, ligand)
        ligand1 = self.Linear_ligand_0(ligand1)
        receptor1, _ = self.cross_attention_block_1(ligand, receptor, receptor)
        receptor1 = self.Linear_receptor_0(receptor1)
        # 第二层 Cross-Attention
        ligand, _ = self.cross_attention_block_2(ligand1, receptor1, ligand1)
        ligand = self.Linear_ligand_1(ligand1)
        receptor, _ = self.cross_attention_block_3(ligand1, receptor1, receptor1)
        receptor = self.Linear_receptor_1(receptor1)
        # 第三层 Cross-Attention                
        ligand1, _ = self.cross_attention_block_4(ligand, receptor, ligand)
        ligand1 = self.Linear_ligand_2(ligand)
        receptor1, _ = self.cross_attention_block_5(ligand, receptor, receptor)
        receptor1 = self.Linear_receptor_2(receptor)
        # 第四层 Cross-Attention                
        ligand, _ = self.cross_attention_block_6(ligand1, receptor1, ligand1)
        ligand = self.Linear_ligand_3(ligand1)
        receptor, _ = self.cross_attention_block_7(ligand1, receptor1, receptor1)
        receptor = self.Linear_receptor_3(receptor1)
        del ligand1, receptor1, ligand
        
        # self-attention
        receptor = self.transformer_encoder(receptor)

        return receptor
    
# 测试CrossAttention_Module。[64,1,256,1152] -> [64,256,2048]
# model = CrossAttention_Module().to('cuda')
# batch_size = 64
# seq_len = 256
# input_dim = 1152
# receptor_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')
# ligand_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')

# output = model(receptor_tensor, ligand_tensor)
# print(output.shape) # [64, 2048]

class Regression_Module(nn.Module):
    '''
    '''
    def __init__(self,
                 input_dim=2048,
                 num_heads=8,
                 num_blocks=6):
        super(Regression_Module, self).__init__()
        # 初始化两个子模型
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        
        self.attention_regression_net = SelfAttention_Module()
        self.cross_attention_module = CrossAttention_Module()
        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=input_dim * 4,
                                                   activation = "relu",
                                                   batch_first = True,
                                                   norm_first = True)
        # 定义 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks // 2)
        self.Grad_module = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks // 2)
        self.softmax = nn.Softmax(dim=1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, input_dim))
        self.regression_rosetta_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 62)
        )
        self.regression_TF_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            self.softmax
        )
        self.regression_kd_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        self.regression_ic50_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

                
    def forward(self, 
                ligand:torch.Tensor, 
                receptor:torch.Tensor, 
                merge:torch.Tensor,
                **kwargs):
        # 先通过 AttentionRegressionNet 模型
        cross_result = self.attention_regression_net(merge)
        # 再通过 CrossAttention_Module 模型
        self_result = self.cross_attention_module(ligand, receptor)
        
        # 在 seqlen 维度上进行拼接
        output = torch.cat([cross_result, self_result], dim=1)
        del cross_result, self_result
        
        # 通过 Transformer 编码器
        output = self.transformer_encoder(output)
        output = self.Grad_module(output)
        output = output.unsqueeze(1)
        output = self.global_avg_pool(output).squeeze(1).squeeze(1).squeeze(1)
        # 最后通过线性层进行回归
        rosetta_output = self.regression_rosetta_layer(output)
        TF_output = self.regression_TF_layer(output)
        kd_output = self.regression_kd_layer(output)
        ic50_output = self.regression_ic50_layer(output)
        output = torch.cat([rosetta_output, TF_output, kd_output, ic50_output], dim=1)
        
        return output
    
    def get_grad_norm_weights(self):
        return nn.ModuleList([self.Grad_module,
                              self.regression_rosetta_layer,
                              self.regression_TF_layer,
                              self.regression_kd_layer,
                              self.regression_ic50_layer])

# #测试Regression_Module。[64,1,256,1152] -> [64,256,2048]
# model = Regression_Module().to('cuda')
# batch_size = 64
# seq_len = 256
# input_dim = 1152
# receptor_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')
# ligand_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')
# merge_tensor = torch.randn(batch_size, 1 ,seq_len, input_dim).to('cuda')

# output = model(ligand_tensor, receptor_tensor, merge_tensor)
# print(output.shape) # [64, 63]
