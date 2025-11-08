import torch
import esm

# 加载预训练的ESM模型
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
layer_num=6
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 关闭dropout以确保结果的确定性

# 准备数据（包括带掩码的序列）
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]

# 转换数据为模型输入格式
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 执行前向传播
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[layer_num], return_contacts=True)

# 提取掩码位置的打分
# 以第二个序列为例，假设掩码位置是第12个位置
mask_token_idx = alphabet.mask_idx
masked_pos = (batch_tokens[1] == mask_token_idx).nonzero(as_tuple=True)[0].item()

# 获取在掩码位置的logits
logits = results['logits'][1, masked_pos]

# 计算掩码位置的概率分布
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# 打印每个氨基酸的概率分布
for i, prob in enumerate(probabilities):
    print(f"{alphabet.get_tok(i)}: {prob.item():.4f}")

# 获取预测的氨基酸
predicted_amino_acid = alphabet.get_tok(probabilities.argmax().item())
print(f"Predicted amino acid at masked position: {predicted_amino_acid}")
