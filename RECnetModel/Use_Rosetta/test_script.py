from pathlib import Path
import os
import json
import os
import subprocess
import shutil
from tqdm import tqdm
import csv
import ast
import torch
from chai_lab.chai1 import run_inference
import numpy as np

# 读取配置文件
with open('/home/users/hcdai/AI-peptide/RunRosetta/config_chai.json', 'r') as f:
    config = json.load(f)

# 打印配置文件内容
print(config)

path = config['path']
parameter = config['parameter']
temp_dir = path['temp']
output_dir=Path(os.path.join(temp_dir, "cif"))
# 初始化循环轮次计数器
loop_count = 3
# 目标序列文件
target_fasta_file = list(Path("/home/users/hcdai/AI-peptide/RunRosetta/test_output_fasta").glob("*.fasta"))  # 使用glob获取所有fasta文件
# print(target_fasta_file[0])
if not target_fasta_file:
    raise FileNotFoundError("没有找到任何fasta文件")

output_dir=Path(os.path.join(temp_dir, "cif"))
output_dir=Path(output_dir)
# 循环读取每个fasta文件
for fasta_file_path in target_fasta_file:
    print(f"Processing {fasta_file_path}")
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
    scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
    # 重命名并移动第一个文件
    if cif_paths:
        first_cif_file = cif_paths[0]
        new_name = f"peptide{loop_count}.cif"
        new_path = os.path.join("/home/users/hcdai/AI-peptide/RunRosetta/test_output_cif", new_name)
        shutil.move(first_cif_file, new_path)
        print(f"Moved and renamed {first_cif_file} to {new_path}")

        # 删除剩下的九个文件
        for cif_file in cif_paths[1:10]:  # 删除后九个文件
            os.remove(cif_file)
            print(f"Deleted {cif_file}")

    # 增加循环轮次计数器
    loop_count += 1
    
# 打印完成信息
print("Done!")