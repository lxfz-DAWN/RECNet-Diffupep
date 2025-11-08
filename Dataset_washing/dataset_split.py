import json
import random
import os

input_file = '/home/users/hcdai/Diffupep/Diffupep_dataset/mask_dataset/60_masked_data_h+v.jsonl'
train_file = 'train.jsonl'
valid_file = 'valid.jsonl'
test_file = 'test.jsonl'

# 第一步：统计总行数
def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for i, _ in enumerate(f, 1):
            pass
    return i

total_lines = count_lines(input_file)
n_train = int(total_lines * 0.8)
n_valid = int(total_lines * 0.1)
n_test = total_lines - n_train - n_valid

# 第二步：生成随机索引并分配
indices = list(range(total_lines))
random.shuffle(indices)
train_indices = set(indices[:n_train])
valid_indices = set(indices[n_train:n_train + n_valid])
test_indices = set(indices[n_train + n_valid:])

# 第三步：按索引写入不同文件
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(train_file, 'w', encoding='utf-8') as ftrain, \
     open(valid_file, 'w', encoding='utf-8') as fvalid, \
     open(test_file, 'w', encoding='utf-8') as ftest:
    for idx, line in enumerate(fin):
        if idx in train_indices:
            ftrain.write(line)
        elif idx in valid_indices:
            fvalid.write(line)
        else:
            ftest.write(line)