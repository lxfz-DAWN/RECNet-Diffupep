import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Hyena操作符基类
class HyenaOperator(nn.Module):
    def __init__(self, dim, order=2, filter_length=3):
        super().__init__()
        self.dim = dim
        self.order = order
        self.filter_length = filter_length
        
        # 输入依赖的参数生成
        self.param_gen = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, order * filter_length)
        )
        
    def forward(self, x):
        B, L, D = x.shape
        params = self.param_gen(x)  # 输入依赖的滤波器参数
        params = params.view(B, L, self.order, self.filter_length)
        
        # 实现Hyena的级联卷积操作
        h = x
        for i in range(self.order):
            kernel = params[..., i, :]
            h = self.causal_conv(h, kernel)
            h = F.gelu(h)
        return h
    
    def causal_conv(self, x, kernel):
        # 实现因果卷积
        pad = (self.filter_length - 1, 0)
        x = F.pad(x, pad, mode='constant')
        return F.conv1d(x.transpose(1,2), kernel.unsqueeze(1)).transpose(1,2)

# StripedHyena块
class StripedHyenaBlock(nn.Module):
    def __init__(self, dim, num_heads, hyena_types=['SE', 'MR', 'LI']):
        super().__init__()
        self.hyenas = nn.ModuleList()
        for ht in hyena_types:
            if ht == 'SE':
                self.hyenas.append(HyenaOperator(dim, order=2, filter_length=7))
            elif ht == 'MR':
                self.hyenas.append(HyenaOperator(dim, order=3, filter_length=128))
            elif ht == 'LI':
                self.hyenas.append(HyenaOperator(dim, order=4, filter_length=256))
        
        # 注意力机制
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, x, rotary_pos_emb):
        # Hyena路径
        residual = x
        for hyena in self.hyenas:
            x = hyena(x)
        x = residual + x
        
        # 注意力路径
        x = self.norm1(x)
        qk = apply_rotary_pos_emb(x, rotary_pos_emb)
        attn_out, _ = self.attn(qk, qk, x)
        x = x + attn_out
        
        # MLP路径
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x

# 旋转位置编码
def apply_rotary_pos_emb(x, pos_emb):
    sin, cos = pos_emb
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)

class Evo2(nn.Module):
    def __init__(self, vocab_size=4, dim=4096, depth=32, num_heads=32, max_seq_len=1_048_576):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            StripedHyenaBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # 旋转位置编码参数
        self.register_buffer('freqs', self._precompute_freqs(dim//2, max_seq_len))
        
        # 输出头
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def _precompute_freqs(self, dim, seq_len):
        theta = 1.0 / (10000 ** torch.linspace(0, 1, dim))
        pos = torch.arange(seq_len)
        freqs = torch.einsum('i,j->ij', pos, theta)
        return torch.polar(torch.ones_like(freqs), freqs)  # 复数形式存储
        
    def forward(self, x):
        B, L = x.shape
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        
        # 应用旋转位置编码
        pos_emb = self.freqs[:L]
        sin = pos_emb.sin()[None, :, None, :]
        cos = pos_emb.cos()[None, :, None, :]
        
        for layer in self.layers:
            x = layer(x, (sin, cos))
            
        return self.head(x)

# 训练框架
class Evo2Trainer:
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        self.optim = optim.AdamW(model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in DataLoader(self.train_data, batch_size=self.config.batch_size):
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                     targets.view(-1),
                                     ignore_index=-1)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()
            
            total_loss += loss.item()
        return total_loss / len(self.train_data)

# 使用示例
config = {
    'dim': 4096,
    'depth': 32,
    'num_heads': 32,
    'vocab_size': 4,  # DNA bases
    'lr': 3e-4,
    'batch_size': 16,
    'seq_len': 8192
}

model = Evo2(**config).cuda()
trainer = Evo2Trainer(model, train_dataset, val_dataset, config)

for epoch in range(100):
    train_loss = trainer.train_epoch()
    print(f"Epoch {epoch} | Loss: {train_loss:.4f}")