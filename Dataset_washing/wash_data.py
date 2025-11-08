import os
import json
import time
import random
import psutil
from datetime import datetime

def log_message(message):
    """记录带时间戳的日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_memory_usage():
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # bytes to MB

def read_fasta_files(folder_path):
    """读取文件夹中的所有FASTA文件路径"""
    fasta_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".fasta", ".fa", ".fna")):
            fasta_files.append(os.path.join(folder_path, file))
    return fasta_files

def parse_fasta(file_path):
    """解析FASTA文件，返回序列列表（去除标题行）"""
    sequences = []
    with open(file_path, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):  # 标题行
                if current_seq:  # 保存前一个序列
                    sequences.append(''.join(current_seq))
                    current_seq = []
            elif line:  # 序列行（跳过空行）
                current_seq.append(line)
        if current_seq:  # 添加最后一个序列
            sequences.append(''.join(current_seq))
    return sequences

def is_nucleic_acid_sequence(seq):
    """检查序列是否为核酸序列（只包含ATCG）"""
    seq = ''.join([char for char in seq if not (char.upper() == 'N' or char.upper() == 'X')])  # 移除N和X
    # DNA_nucleic_chars = set("ATCG")
    # RNA_nucleic_chars = set("AUCG")
    # return all(char.upper() in DNA_nucleic_chars for char in seq) or all(char.upper() in RNA_nucleic_chars for char in seq)
    nucleic_chars = set("ATCGU")
    return all(char.upper() in nucleic_chars for char in seq)

def split_long_sequence(seq, min_len=30, max_len=60):
    """
    将长序列分割为30-128字符的子序列
    算法：使用随机步长的滑动窗口
    """
    sub_sequences = []
    n = len(seq)
    start = 0
    
    while start < n:
        # 随机确定当前子序列长度 (30-128之间)
        seg_len = random.randint(min_len, max_len)
        end = start + seg_len
        
        # 处理序列末尾情况
        if end > n:
            # 确保最后一段至少30个字符
            if n - start >= min_len:
                sub_sequences.append(seq[start:])
            break
        
        sub_sequences.append(seq[start:end])
        
        # 随机确定下一个起始位置 (10-50字符的步长)
        step = random.randint(10, 50)
        start += step
        
        # 处理剩余长度不足的情况
        if n - start < min_len:
            break
    
    return sub_sequences

def process_sequences(sequences):
    """清洗、验证蛋白质序列，并分割长序列"""
    valid_seqs = []
    invalid_count = 0
    split_count = 0
    subseq_count = 0
    
    for seq in sequences:
        # 基本清洗：移除数字、空格和特殊字符
        cleaned = ''.join(filter(str.isalpha, seq))
        cleaned = cleaned.upper()  # 统一转为大写
        
        # 跳过空序列
        if not cleaned:
            invalid_count += 1
            continue
            
        # 过滤核酸序列（只包含ATCG）
        if is_nucleic_acid_sequence(cleaned):
            invalid_count += 1
            continue
            
        seq_len = len(cleaned)
        
        # 处理长序列：分割为子序列
        if seq_len > 60:
            subseqs = split_long_sequence(cleaned)
            split_count += 1
            subseq_count += len(subseqs)
            valid_seqs.extend(subseqs)
        else:
            # 直接保留长度≤60的序列
            valid_seqs.append(cleaned)
            
    return valid_seqs, invalid_count, split_count, subseq_count

def write_json_lines(output_path, sequences):
    """将序列写入JSON Lines格式文件（每行一个JSON对象）"""
    seq_count = 0
    with open(output_path, 'w') as f:
        for seq in sequences:
            json_line = json.dumps({"src": seq, "trg": seq})
            f.write(json_line + '\n')
            seq_count += 1
    return seq_count

def main():
    start_time = time.time()
    log_message("===== 蛋白质序列处理任务开始 =====")
    log_message(f"当前内存占用: {get_memory_usage():.2f} MB")
    
    # 1. 读取FASTA文件
    folder_path = "/home/users/hcdai/Diffupep/Diffupep_dataset/split_uniref50"
    fasta_files = read_fasta_files(folder_path)
    
    if not fasta_files:
        log_message("错误：未找到任何FASTA文件！")
        return
    
    log_message(f"发现 {len(fasta_files)} 个FASTA文件")
    
    # 2. 处理所有文件并准备输出
    output_path = os.path.join("uniref50_sequences.jsonl")
    total_invalid = 0
    total_seqs = 0
    total_splits = 0
    total_subseqs = 0
    total_valid = 0
    
    # 打开输出文件准备写入
    with open(output_path, 'w') as outfile:
        for i, file in enumerate(fasta_files, 1):
            log_message(f"处理文件 {i}/{len(fasta_files)}: {os.path.basename(file)}")
            
            # 解析当前文件
            sequences = parse_fasta(file)
            total_seqs += len(sequences)
            
            # 处理序列
            valid_seqs, invalid_count, split_count, subseq_count = process_sequences(sequences)
            total_invalid += invalid_count
            total_splits += split_count
            total_subseqs += subseq_count
            
            # 写入JSON行
            for seq in valid_seqs:
                json_line = json.dumps({"src": seq, "trg": seq})
                outfile.write(json_line + '\n')
                total_valid += 1
            
            log_message(f"  ├─ 原始序列: {len(sequences)}")
            log_message(f"  ├─ 有效序列: {len(valid_seqs)} | 无效序列: {invalid_count}")
            log_message(f"  ├─ 分割操作: {split_count}次 | 生成子序列: {subseq_count}")
            log_message(f"  └─ 当前内存: {get_memory_usage():.2f} MB")
    
    # 3. 完成处理
    log_message(f"总原始序列数: {total_seqs}")
    log_message(f"总有效序列数: {total_valid} (包含分割产生的子序列)")
    log_message(f"总分割操作: {total_splits}次 | 总生成子序列: {total_subseqs}")
    log_message(f"总过滤序列(核酸/无效): {total_invalid}")
    
    # 性能统计
    duration = time.time() - start_time
    file_size = os.path.getsize(output_path)/(1024**2)
    log_message(f"结果已保存至: {output_path}")
    log_message(f"输出文件大小: {file_size:.2f} MB")
    log_message(f"总耗时: {duration:.2f} 秒")
    log_message(f"峰值内存: {get_memory_usage():.2f} MB")
    log_message("===== 任务完成 =====")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    main()