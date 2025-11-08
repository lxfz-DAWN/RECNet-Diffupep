import os
import glob
import csv
from Bio import SeqIO

# 定义文件路径
output_dir = "/home/users/hcdai/AI-peptide/ChaiRosettaScore/OUTPUT"
ligand_fasta_path = "/home/users/hcdai/AI-peptide/ChaiRosettaScore/ligand/202411051930.fa"
receptor_dir = "/home/users/hcdai/AI-peptide/ChaiRosettaScore/receptor"
csv_output_path = "/home/users/hcdai/AI-peptide/ChaiRosettaScore/output_data.csv"

# 读取多肽药物序列
ligand_sequences = {}
for record in SeqIO.parse(ligand_fasta_path, "fasta"):
    ligand_sequences[record.id] = str(record.seq)

# 读取病毒蛋白序列
receptor_sequences = {}
for fasta_file in os.listdir(receptor_dir):
    if fasta_file.endswith(".fa"):
        virus_name = fasta_file[:-3]  # 去掉.fa后缀
        virus_sequences = {}
        for record in SeqIO.parse(os.path.join(receptor_dir, fasta_file), "fasta"):
            chain_id = record.id.split('_')[-1]  # 提取链ID（A, B, C）
            virus_sequences[chain_id] = str(record.seq)
        # 合并A、B、C链的序列（这里简单地用'|'分隔，也可以选择其他方式）
        combined_sequence = virus_sequences['A']
        receptor_sequences[virus_name] = combined_sequence

# 初始化结果列表
results = []

# 遍历所有pack_input_score.sc文件
for file_path in glob.glob(os.path.join(output_dir, '**', 'pack_input_score.sc'), recursive=True):
    # 获取pack_input_score.sc文件的绝对路径
    pack_abs_path = os.path.abspath(file_path)

    # 获取文件的目录路径
    dir_path = os.path.dirname(pack_abs_path)

    # 获取scores.sc文件的绝对路径
    scores_abs_path = os.path.join(dir_path,'scores.sc')
    
    # 提取多肽药物名称和病毒蛋白名称
    parts = dir_path.split(os.sep)
    peptide_drug_name = parts[-2]
    virus_protein_full_name = parts[-1]
    virus_protein_name = virus_protein_full_name.split('_')[0]  # 取_A前的部分

    # 获取多肽药物序列和病毒蛋白序列
    ligand_sequence = ligand_sequences.get(peptide_drug_name, "Sequence Not Found")
    virus_sequence = receptor_sequences.get(virus_protein_name, "Sequence Not Found")
    
    # 读取pack_input_score.sc打分数据
    with open(pack_abs_path, 'r') as sc_file:
        lines = sc_file.readlines()
        if len(lines) >= 3:
            headers = [f"pack_{header}" for header in lines[1].strip().split()[1:]]  # 第二行为表头，去掉第一个，并添加前缀
            data = lines[2].strip().split()     # 第三行为数据
            # 去掉第一个表头和第一个数据
            data = data[1:]
            # 创建字典存储打分数据
            pack_score_data = dict(zip(headers, data))
            

        # 获取scores.sc打分数据
    with open(scores_abs_path, 'r') as sc_file:
        lines = sc_file.readlines()
        if len(lines) >= 3:
            headers = [f"scores_{header}" for header in lines[1].strip().split()[1:]]  # 第二行为表头，去掉第一个，并添加前缀
        # 初始化一个字典来存储数据，键为表头，值为空列表
        scores_data = {header: [] for header in headers}
        
        # 从第三行开始读取数据（索引为2的行）
        for line in lines[2:]:
            # 分割每行的数据，并跳过第一个数据（索引为0的元素）
            data = line.strip().split()[1:]

                # 检查数据长度是否与表头长度一致
            if len(data) != len(headers):
                raise ValueError(f"数据长度 {len(data)} 与表头长度 {len(headers)} 不一致")
            
            # 将数据添加到对应的表头列表中
            for i, header in enumerate(headers):
                scores_data[header].append(data[i])
            
    merged_scores_data = {**scores_data, **pack_score_data}
    # 将结果添加到列表中
    results.append({
        'ligand_name': peptide_drug_name,
        'ligand_sequence': ligand_sequence,
        'receptor_name': virus_protein_name,
        'receptor_sequence': virus_sequence,
        **merged_scores_data  # 将打分数据作为额外字段添加
        })
        
# 将结果写入CSV文件
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['ligand_name', 'ligand_sequence', 'receptor_name', 'receptor_sequence']
    # 添加打分数据的表头
    for key in results[0].keys():
        if key not in fieldnames:
            fieldnames.append(key)
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
 
print(f"Results have been written to {csv_output_path}")