import json

def merge_json_lines(file1, file2, output_file):
    """按行合并两个JSONL文件（每行一个JSON对象）"""
    merged_data = []
    
    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip()
            if line:  # 跳过空行
                data = json.loads(line)
                merged_data.append(data)
    
    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            if line:  # 跳过空行
                data = json.loads(line)
                merged_data.append(data)
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in merged_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成！共 {len(merged_data)} 条数据")
    return merged_data

# 使用示例
merge_json_lines('/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/Diffupep_trasn/Diffupep/DiffuSeq-diffuseq-v2/datasets/uniref50-all-better2better/aujsdhbakshjdhask.jsonl', '/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/Diffupep_trasn/Diffupep/DiffuSeq-diffuseq-v2/datasets/uniref50-all-better2better/train.jsonl', 'train.json')