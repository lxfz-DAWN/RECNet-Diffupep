from pathlib import Path
import os
import json
import os
import subprocess
import shutil
from tqdm import tqdm
import csv
import ast
# 读取配置文件
with open('/home/users/hcdai/AI-peptide/RunRosetta/config_chai.json', 'r') as f:
    config = json.load(f)

# 打印配置文件内容
print(config)

path = config['path']
parameter = config['parameter']


def convert_cif_to_pdb(cif_file_path,pdb_file_path):
    # 读取cif文件转换为pdb文件
    if not os.path.exists(pdb_file_path):
        os.makedirs(pdb_file_path)
    # 定义输入和输出目录
    input_cif_directory = cif_file_path  
    output_pdb_directory = pdb_file_path  

    # 获取输入目录中所有CIF文件的列表并进行排序
    cif_files = [f for f in os.listdir(input_cif_directory) if f.endswith(".cif")]
    cif_files.sort()  # 按字母顺序排序

    # 使用tqdm为循环添加进度条
    for cif_file in tqdm(cif_files, desc="Converting files"):
        cif_path = os.path.join(input_cif_directory, cif_file)
        pdb_file = cif_file.replace(".cif", ".pdb")
        pdb_path = os.path.join(output_pdb_directory, pdb_file)

        obabel_path = path["obabel"]
        # 使用Open Babel将CIF转换为PDB
        obabel_command = [obabel_path, cif_path, "-O", pdb_path]
        with open(os.devnull, 'w') as devnull:
            subprocess.run(obabel_command, stdout=devnull, stderr=devnull)

        # 使用awk删除文件头
        awk_command = ["awk", '/^ATOM |^TER/']
        with open(pdb_path, "rb") as input_file:
            awk_process = subprocess.run(awk_command, stdin=input_file, stdout=subprocess.PIPE, text=True)
            tmp2_content = awk_process.stdout

        with open(pdb_path, "w") as tmp2_file:
            tmp2_file.write(tmp2_content)

    print("转换完成")
    
cif_file_path = "/home/users/hcdai/AI-peptide/RunRosetta/test_output_cif"
pdb_file_path = "/home/users/hcdai/AI-peptide/RunRosetta/test_output_pdb"
convert_cif_to_pdb(cif_file_path,pdb_file_path)