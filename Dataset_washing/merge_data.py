import json
import os
import random

def merge_jsonl_files_advanced(folder_path=None, splits=None, output_names=None, shuffle=True, seed=42):
    """
    高级版本的JSON Lines文件合并函数
    
    参数:
    folder_path: JSON文件所在文件夹路径
    splits: 分割比例列表，如 [40, 6, 6]
    output_names: 输出文件名列表
    shuffle: 是否随机打乱文件
    seed: 随机种子
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 设置默认参数
    if folder_path is None:
        folder_path = "/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/dataset_washing/mutation_compare/dataset_to_wash"
    
    if splits is None:
        splits = [40, 6, 6]
    
    if output_names is None:
        output_names = ['merged_train.json', 'merged_val.json', 'merged_test.json']
    
    # 获取所有json文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # 过滤掉输出文件，避免重复包含
    json_files = [f for f in json_files if f not in output_names]
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 检查文件数量是否匹配
    total_splits = sum(splits)
    if len(json_files) != total_splits:
        print(f"警告：文件数量 ({len(json_files)}) 与分割总数 ({total_splits}) 不匹配")
        # 调整分割比例以适应实际文件数量
        if len(json_files) < total_splits:
            splits = splits[:]  # 创建副本
            splits[-1] = len(json_files) - sum(splits[:-1])
    
    # 随机打乱文件列表
    if shuffle:
        random.shuffle(json_files)
    
    # 按比例分割文件
    split_files = []
    start = 0
    for split_size in splits:
        end = start + split_size
        split_files.append(json_files[start:end])
        start = end
    
    # 合并文件的函数 - 逐行处理
    def merge_files(file_list, output_name):
        total_lines = 0
        output_path = os.path.join(folder_path, output_name)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for filename in file_list:
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        file_lines = 0
                        for line in infile:
                            line = line.strip()
                            if line:  # 跳过空行
                                # 可选：验证JSON格式
                                try:
                                    json.loads(line)
                                    outfile.write(line + '\n')
                                    file_lines += 1
                                    total_lines += 1
                                except json.JSONDecodeError:
                                    print(f"  警告：文件 {filename} 中第 {file_lines+1} 行包含无效的JSON格式")
                        
                        print(f"  已处理 {filename}: {file_lines} 行")
                        
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")
        
        print(f"已创建 {output_name}，包含 {total_lines} 行数据（来自 {len(file_list)} 个文件）")
        return total_lines
    
    # 合并所有分割的数据集
    total_records = 0
    split_info = []
    
    for i, (files, output_name) in enumerate(zip(split_files, output_names)):
        print(f"\n开始合并分割 {i+1} ({len(files)} 个文件) -> {output_name}")
        records_count = merge_files(files, output_name)
        total_records += records_count
        split_info.append((len(files), records_count, output_name))
    
    # 打印详细报告
    print("\n" + "="*50)
    print("合并完成报告：")
    print("="*50)
    for i, (file_count, record_count, output_name) in enumerate(split_info):
        print(f"分割 {i+1}: {file_count} 个文件 -> {output_name} ({record_count} 行记录)")
    
    print(f"\n总计: {total_records} 行记录")
    print(f"预计总数据量: {total_records * 400000 // len(json_files)} 行 (基于平均文件大小)")

# 使用方法
if __name__ == "__main__":
    # 使用默认参数
    merge_jsonl_files_advanced()
    
    # 或者使用自定义参数
    # merge_jsonl_files_advanced(
    #     splits=[30, 11, 11],  # 不同的分割比例
    #     output_names=['train.json', 'validation.json', 'test.json'],
    #     seed=123  # 固定随机种子确保可重复性
    # )