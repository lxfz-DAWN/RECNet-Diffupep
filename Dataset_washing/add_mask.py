import json
import random

def mask_sequence(seq, mask_token='<mask>', mask_ratio=0.15):
    seq = list(seq)
    length = len(seq)
    num_mask = max(1, int(length * mask_ratio))
    mask_indices = random.sample(range(length), num_mask)
    for idx in mask_indices:
        seq[idx] = mask_token
    return ''.join(seq)

input_file = '/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/dataset_washing/spilt_sequence/merged_val.json'   # 原始jsonl文件名
output_file = 'merged_val_masked_data.jsonl'  # 输出jsonl文件名

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        masked_src = mask_sequence(data['src'])
        data['src'] = masked_src
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')