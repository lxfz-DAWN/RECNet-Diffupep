import json
import random

file_path = "/home/users/hcdai/Diffupep/Diffupep_dataset/mask_dataset/test.jsonl"
output_file = 'processed_data.jsonl'
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
    print(f"已读取文件: {file_path}, 总数据量: {len(data)} 条")

process_data = []
other = set("ATCGUNX") 

min_len = 30
max_len = 60

for item in data:
    src , trg = item['src'], item['trg']

    if len(trg) < min_len :
        continue

    if all(char.upper() in other for char in trg):
        continue

    if min_len <= len(trg) <= max_len:
        process_data.append(item)
        continue

    if len(trg) > max_len:
        start = 0
        n = len(trg)
        src = src.replace('<mask>', '0')

        while start < n:
            seg_len = random.randint(min_len, max_len)
            end = start + seg_len

            if end > n:
                if n - start >= min_len:
                    process_data.append({'src':src[start:].replace('0', '<mask>'),'trg':trg[start:]})
                break

            process_data.append({'src':src[start:end].replace('0', '<mask>'),'trg':trg[start:end]})
            step = random.randint(10, 50)
            start += step

            if n - start < min_len:
                break

with open(output_file, 'w', encoding='utf-8') as out:
    for item in process_data:
        out.write(json.dumps(item) + '\n')

print(f"处理完成，已保存到: {output_file}, 总数据量: {len(process_data)} 条")
