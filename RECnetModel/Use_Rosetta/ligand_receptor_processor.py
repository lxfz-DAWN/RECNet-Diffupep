import csv
import json
import os
import subprocess
import shutil
import datetime
import csv


# Load the config JSON file
config_path = "/home/users/hcdai/AI-peptide/RunRosetta/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
workspace = config["workspace"]
data_csv = config["ligand_receptor_data"]
out_put_dir = config["pdb_index"]

os.chdir(workspace)
# print(data_csv)


def ligand_receptor_processor(headers):
    
    ligand_receptor_list = []
        
        # 遍历CSV文件的每一行
    for row in csv_reader:
        # 提取指定列的数据并添加到列表       
        ligand_receptor_list.append(row[headers[2]])
            
    # 返回提取的列数据列表
    return ligand_receptor_list


def ligand_receptor_seperator(ligand_receptor_list):
    ligands_list = [item.split(';')[0] for item in ligand_receptor_list]
    receptors_list = [item.split(';')[1].rstrip(';') for item in ligand_receptor_list]
    receptors_list = [element.lstrip() for element in receptors_list]
    ligand_receptor_dict = {'ligand': ligands_list,'receptor': receptors_list}
    return ligand_receptor_dict

def name_processor(headers):
    
    name_list = []
    
    # 遍历CSV文件的每一行
    for row in csv_reader:
        name_list.append(row[headers[0]])
        
    name_list = [element.rstrip() for element in name_list]
        
        
    # 返回name_list
    return name_list

# def write_lists_to_csv(pdb_list, receptor_chain_list, ligand_chain_list, file_path):
#     # 定义表头
#     headers = ["PDB", "receptor_chain", "ligand_chain"]
    
#     # 检查三个列表的长度是否一致，如果不一致，可以打印警告或抛出异常
#     if len(pdb_list) != len(receptor_chain_list) or len(pdb_list) != len(ligand_chain_list):
#         print("警告：三个数据列表的长度不一致，数据可能会被截断或丢失。")
#         # 为了避免错误，我们可以截断较长的列表以匹配最短的列表
#         max_length = min(len(pdb_list), len(receptor_chain_list), len(ligand_chain_list))
#         pdb_list = pdb_list[:max_length]
#         receptor_chain_list = receptor_chain_list[:max_length]
#         ligand_chain_list = ligand_chain_list[:max_length]
    
#     # 准备要写入CSV文件的数据行
#     rows = zip(pdb_list, receptor_chain_list, ligand_chain_list)
    
#     # 写入CSV文件
#     with open(file_path, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(headers)  # 写入表头
#         writer.writerows(rows)    # 写入数据行
    
    # print(f"数据已成功写入 {file_path}")
 
    
    

with open(data_csv, mode='r', newline='', encoding='utf-8') as csv_file:
        # 创建csv.reader对象
        csv_reader = csv.DictReader(csv_file)
        # 获取表头（列名），这里只是为了确认csv.DictReader工作正常，实际代码中可以不用
        headers = csv_reader.fieldnames
        ligand_receptor_list = ligand_receptor_processor(headers)
        ligand_receptor_dict = ligand_receptor_seperator(ligand_receptor_list)
        receptors_list = ligand_receptor_dict['receptor']
        ligands_list = ligand_receptor_dict['ligand']
        name_list = name_processor(headers)
        # write_lists_to_csv(name_list, receptors_list, ligands_list, out_put_dir)
# print(headers)
# print(ligand_receptor_list)
print(len(name_list))
# print(ligand_receptor_dict)
