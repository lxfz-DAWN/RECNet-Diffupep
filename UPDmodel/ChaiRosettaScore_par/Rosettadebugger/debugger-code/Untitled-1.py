# %%
from pathlib import Path
import os
import json
import subprocess
import shutil
import logging
import logging.config
import csv
import datetime
from tqdm import tqdm

# 读取配置文件
with open('/home/users/hcdai/AI-peptide/ChaiRosettaScore_par/config3.json', 'r') as f:
    config = json.load(f)
    logging.info(f"Config file loaded: {config}")
    # # 打印配置文件内容
    # print(config)
path = config['path']
parameter = config['parameter']

# %% [markdown]
# ## 这段是负责Rosetta打分的函数

# %%
def rosetta_score(pdb_file_path, output_dir, rosetta_path:dict = path["rosetta"]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    
    print("Running Rosetta scoring...")
    

    # 执行命令的函数
    def run_command(command):
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    
    rosetta_temp_path = os.path.join(path["temp"], "rosetta")
    if not os.path.exists(rosetta_temp_path):
        os.makedirs(rosetta_temp_path)
    scorefile_path = os.path.join(output_dir, "scores.sc")
    if os.path.exists(scorefile_path):
        os.remove(scorefile_path)

    ori_path = os.getcwd()
    # 下策：改变工作路径
    os.chdir(rosetta_temp_path)

    # 对输入文件进行打分
    score_command = f"{rosetta_path['score_executable']} -s {pdb_file_path} -no_optH false -ignore_unrecognized_res -out:pdb"
    print(score_command)
    run_command(score_command)
    print("Rosetta score_command successful.")

    # 将rosetta_temp_path中打分后的文件重命名，为原名称后面加上"_scored"
    pdb_files:list = [f for f in os.listdir(rosetta_temp_path) if f.endswith(".pdb")] # type: ignore
    for pdb_file in pdb_files:
        os.rename(os.path.join(rosetta_temp_path, pdb_file),
                   os.path.join(rosetta_temp_path, 
                                pdb_file.replace(".pdb", "_scored.pdb")))    
        
    print("rename successful.")
    print(output_dir)

    # 从rosetta_temp_path中逐条读取文件相对路径，进行接口分析，分析结果保存至output_dir中下以ligand_name命名的文件夹中
    for pdb_file in pdb_files:
        scored_pdb_file = pdb_file.replace(".pdb", "_scored.pdb")
        scored_pdb_path = os.path.join(rosetta_temp_path, scored_pdb_file)
        
    # 进行接口分析，输出结果到指定的文件夹
    analyze_command = f"{rosetta_path['InterfaceAnalyzer']} -s {scored_pdb_path} -fixedchains A B C@{rosetta_path['pack_input_options']}"
    run_command(analyze_command)

    print(f"InterfaceAnalyzer successful for {scored_pdb_file}.")

    # 复制temp文件夹内的所有文件文件到output_dir
    for file in os.listdir(rosetta_temp_path):
        # if file.endswith(".pdb") or file.endswith(".sc"):
        if file.endswith(".sc"):
            shutil.copy(os.path.join(rosetta_temp_path, file), os.path.join(output_dir, file))
        

    # 下策：重新定义回旧工作路径    
    os.chdir(ori_path)

    print("Rosetta scoring successful.")

pdb_path = "/home/users/hcdai/AI-peptide/ChaiRosettaScore_par/temp/temp3/pdb/pred.model_idx_0.pdb"
score_output_dir = "/home/users/hcdai/AI-peptide/ChaiRosettaScore/Rosettadebugger/Rosettadebugger-Output"
rosetta_path = path["rosetta"]
rosetta_score(pdb_path, score_output_dir, rosetta_path)
## 运行发现这个pdb确实无法被这块代码运行

#/home/users/hcdai/miniconda3/envs/Chai-1/bin/python /home/users/hcdai/AI-peptide/ChaiRosettaScore_par/Rosettadebugger/debugger-code/Untitled-1.py