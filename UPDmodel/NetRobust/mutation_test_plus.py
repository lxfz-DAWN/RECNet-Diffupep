import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import esm
import random
import os

# 加载esm2大模型和字典
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
layer_num = 6
model.train()  # 设置为训练模式以进行微调
batch_converter = alphabet.get_batch_converter()

def mutate_sequence(sequence, action):
    sequence = list(sequence)
    
    if action == 0:  # substitution
        idx = random.randint(0, len(sequence) - 1)
        original_aa = sequence[idx]
        possible_mutations = [aa for aa in alphabet.standard_toks[1:-1] if aa != original_aa]
        sequence[idx] = random.choice(possible_mutations)
    
    elif action == 1:  # deletion
        if len(sequence) > 1:
            idx = random.randint(0, len(sequence) - 1)
            sequence.pop(idx)
    
    elif action == 2:  # duplication
        idx = random.randint(0, len(sequence) - 1)
        sequence.insert(idx, sequence[idx])
    
    elif action == 3:  # inversion
        if len(sequence) > 1:
            start = random.randint(0, len(sequence) - 2)
            end = random.randint(start + 1, len(sequence) - 1)
            sequence[start:end+1] = reversed(sequence[start:end+1])
    
    return ''.join(sequence)

def score_sequence(sequence):
    data = [("sequence", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer_num], return_contacts=False)
    token_representations = results["representations"][layer_num]

    mean_score = token_representations.mean().item()
    return mean_score

class PolicyNetwork(nn.Module):
    def __init__(self, esm_model, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.esm_model = esm_model
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, batch_tokens):
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[layer_num], return_contacts=False)
        token_representations = results["representations"][layer_num]
        x = token_representations.mean(dim=1)  # 平均所有token的表示
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

def reinforce(sequence, policy_net, optimizer, generations=10, num_mutations=1, gamma=0.99):
    current_sequence = sequence
    current_score = score_sequence(current_sequence)
    rewards = []
    log_probs = []

    for generation in range(generations):
        for _ in range(num_mutations):
            # 获取动作概率分布
            data = [("sequence", current_sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            action_probs = policy_net(batch_tokens)
            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            
            # 执行动作
            mutated_sequence = mutate_sequence(current_sequence, action.item())
            mutated_score = score_sequence(mutated_sequence)
            
            rewards.append(mutated_score)
            log_probs.append(log_prob)
            
            # 更新当前序列和分数
            if mutated_score > current_score:
                current_sequence = mutated_sequence
                current_score = mutated_score

        print(f"Generation {generation + 1}: Sequence {current_sequence}, Score {current_score}")

    # 计算损失并更新策略网络
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = sum([r * (gamma ** i) for i, r in enumerate(rewards[t:])])
        discounted_rewards.append(Gt)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    loss = torch.sum(-torch.stack(log_probs) * discounted_rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return current_sequence, current_score

def save_policy_network(policy_net, optimizer, path="policy_net.pth"):
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Policy network saved to {path}")

def load_policy_network(esm_model, input_size, hidden_size, path="policy_net.pth"):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        policy_net = PolicyNetwork(esm_model, input_size, hidden_size, output_size=4)
        optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Policy network loaded from {path}")
        return policy_net, optimizer
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")

# 初始化策略网络，隐藏层大小和输出大小与之前一致
hidden_size = 128
input_size = 320  # 这个值根据实际的esm2模型输出大小确定
policy_net = PolicyNetwork(model, input_size, hidden_size, output_size=4)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# 加载之前保存的策略网络（如果存在）
# policy_net, optimizer = load_policy_network(model, input_size, hidden_size, "policy_net.pth")

# 示例初始序列
initial_sequence = "ACDEFGHIKLMNPQRSTVWY"

# 进行强化学习序列进化
final_sequence, final_score = reinforce(initial_sequence, policy_net, optimizer, generations=20, num_mutations=2)
print(f"Final Sequence: {final_sequence}, Final Score: {final_score}")

# 保存策略网络
save_policy_network(policy_net, optimizer, path="policy_net.pth")
