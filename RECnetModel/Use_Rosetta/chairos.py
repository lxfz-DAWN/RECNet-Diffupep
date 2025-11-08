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
# # 设定工作区路径
# os.chdir(path["workspace"])

# getcwd = os.getcwd()
# print("Current working directory: ", getcwd)
    
def target_sequence_generator(temp_dir,chaiworkdir):
    
    output_dir=Path(os.path.join(temp_dir, "cif"))
    # 目标序列文件
    target_fasta_file = list(Path("/home/users/hcdai/AI-peptide/RunRosetta/test_output_fasta").glob("*.fasta"))  # 使用glob获取所有fasta文件
    print(target_fasta_file[0])
    if not target_fasta_file:
        raise FileNotFoundError("没有找到任何fasta文件")
    # 目标脚本
    script = f'''from pathlib import Path
import numpy as np
import torch
import os
from chai_lab.chai1 import run_inference
output_dir=Path(output_dir)
# 循环读取每个fasta文件
for fasta_file_path in target_fasta_file:
    candidates = run_inference(
        fasta_file=Path(fasta_file_path),
        output_dir=output_dir,
        # 'default' setup
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=42,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )
    cif_paths = candidates.cif_paths
    scores = [rd.aggregate_score for rd in candidates.ranking_data]
    # Load pTM, ipTM, pLDDTs and clash scores for sample 2
    scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))'''  
    
    # 将目标脚本写入chaiworkdir下的run.py文件
    run_file = Path(os.path.join(chaiworkdir, "run.py"))
    with open(run_file, 'w') as f:
        f.write(script)

    # 成功写入脚本文件
    print(f"Chai script generated at {run_file}.")
    # 执行脚本
    subprocess.run(["python", str(run_file)], check=True)


    return run_file, output_dir, target_fasta_file

# 定义运行chai脚本函数
def run_chai_script(chai_python_path,run_file):
    # # 使用subprocess运行chai脚本
    # run_file = str(run_file)
    # command = [chai_python_path, run_file]
    # print("command:", command)    
    # process = subprocess.run(command,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # # 打印输出和错误信息
    # print("输出:", process.stdout.decode())
    # print("错误:", process.stderr.decode())

    # 使用os.system运行chai脚本
    run_file = str(run_file)
    command = f"{chai_python_path} {run_file}"
    os.system(command)

    return True

# 定义运行bash脚本函数
def run_bash_script(bash_script_path):
    # 使用subprocess运行bash脚本
    bash_script_path = str(bash_script_path)

    return True

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
    
pdb_file_path = "/home/users/hcdai/AI-peptide/RunRosetta/test_output_pdb"
cif_file_path = "/home/users/hcdai/AI-peptide/RunRosetta/temp/cif"
temp_dir= path["temp"]
chaiworkdir= "/home/users/hcdai/AI-peptide/Chai-1/chai-lab/RMSD_Time-test"
chai_python_path= "/home/users/hcdai/miniconda3/envs/Chai-1/bin/python"
run_file, target_fasta_file, cif_output_dir = target_sequence_generator(temp_dir,chaiworkdir)
print(cif_output_dir)
run_chai_script(chai_python_path, run_file)
convert_cif_to_pdb(cif_file_path, pdb_file_path)