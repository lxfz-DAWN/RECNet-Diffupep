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


# 定义日志配置文件
LOGGING_CONFIG = {
    "version": 1, 
    "disable_existing_loggers": False, 
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }, 
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": f"{os.getcwd()}/ChaiRosettaScore.log",
            "encoding": "utf8",
            "mode": "a",
        }
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console", "file"],
        }
    }
}

# 配置日志系统
logging.config.dictConfig(LOGGING_CONFIG)

# 开始日志记录
logging.info("ChaiRosettaScore started.")

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)
    logging.info(f"Config file loaded: {config}")
    # # 打印配置文件内容
    # print(config)

path = config['path']
parameter = config['parameter']
# 设定工作区路径
os.chdir(path["workspace"])

getcwd = os.getcwd()
print("Current working directory: ", getcwd)
logging.info(f"Current working directory: {getcwd}")

# %%
def parse_fasta(fasta_file = None):  
    """  
    解析FASTA文件，并将序列名称和对应的氨基酸序列（字符串）保存为字典。  
      
    参数:  
    fasta_file (str): fasta文件的路径。  
      
    返回:  
    dict：词典，键值对表示fasta文件中的序列名称和氨基酸，键为序列名称，值为氨基酸序列（字符串形式）。  
    """  
    
    if fasta_file is None:  
        raise ValueError("请输入fasta文件路径！")  
    
    sequences = {}  
    current_name = None  
    current_sequence = []  
      
    with open(fasta_file, 'r') as file:  
        for line in file:  
            line = line.strip()  # 去除行尾的换行符和可能的空格  
            if line.startswith('>'):  # 判断是否为序列名称行  
                if current_name is not None:  
                    # 如果之前已经收集到序列，则保存到字典中  
                    sequences[current_name] = current_sequence  
                current_name = line[1:]  # 去除开头的'>'  
                current_sequence = []  # 重置当前序列  
            elif line:  # 忽略空行  
                current_sequence.extend(line)  # 将当前行添加到当前序列中  
      
    # 将最后一个序列添加到字典中  
    if current_name is not None:  
        sequences[current_name] = current_sequence

    # 新增功能输出的序列为字符串，没有逗号分割
    for key in sequences:
        sequences[key] = ''.join(sequences[key])

    return sequences



# %%
# receptor部分处理
# 读取receptor文件夹下的fasta文件,计算文件数目，并分别将文件内容保存至字典中，以文件名为key，文件内容为value。
# receptor的文件一个文件只能有一条序列！！！

receptor_dir = path["receptor_database"]

receptor_file_list = os.listdir(receptor_dir)
receptor_file_num = len(receptor_file_list)

receptors_dict = {} # 该字典首先获取receptor文件名，然后读取文件内容，并保存至字典中

for receptor_file in receptor_file_list:
    # 分析receptor_file文件内容，将文件中的序列保存至字典中
    receptors_dict[receptor_file] = parse_fasta(os.path.join(receptor_dir, receptor_file))

logging.info(f"Current working directory: {receptors_dict.keys()}")

# ligand部分处理
# 读取ligand文件夹下的fasta文件,计算文件数目，并定义生成器读取文件内容
# ligand文件能有条序列，文件夹中可以有多个文件，所以定义生成器逐条读取文件中fasta序列

ligand_dir = path["ligand_database"]

ligand_file_list = os.listdir(ligand_dir)
ligand_file_num = len(ligand_file_list)

# %%
def ligand_generator():
    '''配体序列生成器，用于ligand文件夹中读取所有配体文件'''
    for ligand_file in ligand_file_list:
        sequences = parse_fasta(os.path.join(ligand_dir, ligand_file))   
        # 逐条读取sequences中的key和value，使用生成器输出
        for key, value in sequences.items():
            yield key, value

def convert_to_protein_format(key, value):
    '''标准化序列格式'''
    key = f">protein|name={key}"
    value = value.upper()
    return key, value

# 定义目标序列生成函数
def target_sequence_generator(receptor,ligand_name,ligand_sequence,receptor_chain_num:int = 3):
    '''定义目标序列生成函数，输入为receptor序列，ligand名称和序列，输出为目标序列
    目标序列为receptor*receptor_chain_num+ligand'''
    target_sequence:str = str(receptor * int(receptor_chain_num) + ligand_name + "\n" + ligand_sequence)
    return target_sequence

# 生成chai运行python脚本
def generate_chai_script(ligand_name,target_sequence,chaiworkdir,tempdir=path["temp"]):
    # 生成chai运行python脚本
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    if not os.path.exists(chaiworkdir):
        os.makedirs(chaiworkdir)

    # 将目标序列写入temp下的fasta文件
    target_fasta_file = Path(os.path.join(tempdir, f"{ligand_name}.fasta"))
    with open(target_fasta_file, 'w') as f:
        f.write(f"{target_sequence.strip()}")
    output_dir = Path(os.path.join(tempdir, "cif"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 目标脚本
    script = f'''from pathlib import Path
import numpy as np
import torch
import os
from chai_lab.chai1 import run_inference
output_dir=Path("{output_dir}")
candidates = run_inference(
    fasta_file=Path("{target_fasta_file}"),
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

    return run_file, target_fasta_file, output_dir
    
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
    

# 定义读取cif文件转换为pdb文件的函数
def convert_cif_to_pdb(cif_file_path,pdb_file_path):
    # 读取cif文件转换为pdb文件
    if not os.path.exists(pdb_file_path):
        os.makedirs(pdb_file_path)
    # 定义输入和输出目录
    input_cif_directory = cif_file_path  
    output_pdb_directory = pdb_file_path  

    # 获取输入目录中所有CIF文件的列表并进行排序
    cif_files = [f for f in os.listdir(input_cif_directory) if f.endswith("0.cif")]
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

# 使用rosetta进行打分

# def rosetta_score(pdb_file_path, output_dir, rosetta_path:dict = path["rosetta"]):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)  
    
#     print("Running Rosetta scoring...")

#     # 执行命令的函数
#     def run_command(command):
#         result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#         return result.stdout
    
#     rosetta_temp_path = os.path.join(path["temp"], "rosetta")
#     if not os.path.exists(rosetta_temp_path):
#         os.makedirs(rosetta_temp_path)

#     # 对输入文件进行打分
#     score_command = f"{rosetta_path['score_executable']} -s {pdb_file_path}/*.pdb -no_optH false -ignore_unrecognized_res -out:path {rosetta_temp_path}"
#     print(score_command)
#     run_command(score_command)
#     print("Rosetta score_command successful.")

#     # 将rosetta_temp_path中打分后的文件重命名，为原名称后面加上"_scored"
#     pdb_files:list = [f for f in os.listdir(rosetta_temp_path) if f.endswith(".pdb")] # type: ignore
#     for pdb_file in pdb_files:
#         os.rename(os.path.join(rosetta_temp_path, pdb_file),
#                    os.path.join(rosetta_temp_path, 
#                                 pdb_file.replace(".pdb", "_scored.pdb")))    
        
#     print("rename successful.")
    
#     # 从pdb_file_path中逐条读取文件相对路径，进行接口分析，分析结果保存至output_dir中下以ligand_name命名的文件夹中
#     for pdb_file in pdb_files:
#         scored_pdb_file = pdb_file.replace(".pdb", "_scored.pdb")
#         scored_pdb_path = os.path.join(rosetta_temp_path, scored_pdb_file)
        
#         # 进行接口分析，输出结果到指定的文件夹
#         analyze_command = f"{rosetta_path['InterfaceAnalyzer']} -s {scored_pdb_path} -fixedchains A B @{rosetta_path['pack_input_options']} -out:path {output_dir}"
#         run_command(analyze_command)

#         print(f"InterfaceAnalyzer successful for {scored_pdb_file}.")

# 测试rosetta score
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
    # score_command = f"{rosetta_path['score_executable']} -s {pdb_file_path}/*.pdb -no_optH false -ignore_unrecognized_res -out:file:scorefile {scorefile_path} -out:pdb"
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

 
# 定义运行主函数
# 输入参数receptor_dict，ligand_name, ligand_sequence, receptor_chain_num,path,paraparamete

def chai_rosetta_score(receptor:dict, ligand_name:str, ligand_sequence:str, receptor_chain_num = parameter["receptor_chain_num"], path = path):
    '''主函数，从目标文件中读取配体和受体，运行后输出打分文件
    Args:
        receptor: 字典，包含受体名称和序列
        ligand_name: 配体名称
        ligand_sequence: 配体序列
        receptor_chain_num: 受体链数
        path: 字典，包含路径信息
        paraparamete: 字典，包含参数信息
    '''

    # 提取chai和rosetta路径
    chai_path = path["chai-1"]
    rosetta_path = path["rosetta"]

    # 读取受体字典中的序列，使得受体名为变量，序列为值
    # 统计受体字典的key数目
    receptor_num = len(receptor)
    if receptor_num == 0:
        print("Error: No receptor found in the receptor dictionary.")
        return None
    elif receptor_num == 1:
        receptor_name_ori = list(receptor.keys())[0]
        receptor_sequence = list(receptor.values())[0]
        receptor_name, receptor_sequence = convert_to_protein_format(receptor_name_ori, receptor_sequence)
        receptor_last = str(receptor_name + "\n" + receptor_sequence + "\n")
        receptor_chain_num = receptor_chain_num
    else:
        receptor_name_ori = ""
        receptor_last = ""
        for key, value in receptor.items():
            receptor_name_ori += key + "_"
            key, value = convert_to_protein_format(key, value)
            receptor_last += str(key + "\n" + value + "\n")
        # 去除末尾的下划线
        receptor_name_ori = receptor_name_ori[:-1]
        
        receptor_chain_num = 1

    # 标准化ligand序列格式
    ligand_name_ori = ligand_name
    ligand_name, ligand_sequence = convert_to_protein_format(ligand_name, ligand_sequence)

    # 定义目标序列生成函数
    target_sequence = target_sequence_generator(receptor_last, ligand_name, ligand_sequence, receptor_chain_num)

    # 生成chai运行python脚本
    run_file, target_fasta_file, cif_output_dir = generate_chai_script(ligand_name, target_sequence,chai_path["chaiworkdir"],tempdir=path["temp"])

    # 运行chai脚本
    run_chai_script(chai_path["chai_python_executable"], run_file)
    # run_bash_script(chai_path["ChaiRunPython"])

    pdb_path = os.path.join(path["temp"], "pdb")
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path)
    # 读取cif文件转换为pdb文件
    convert_cif_to_pdb(cif_output_dir, pdb_path)    

    # 运行rosetta进行打分,并输出结果文件至output_dir中以ligand_name命名的文件夹中的receptor_name_ori命名的文件夹中
    score_output_dir = os.path.join(path["output"], ligand_name_ori, receptor_name_ori)
    if not os.path.exists(score_output_dir):
        os.makedirs(score_output_dir)
    pdb_names:list = [f for f in os.listdir(pdb_path) if f.endswith(".pdb")]
    print(pdb_names)
    # rosetta_score(pdb_path, score_output_dir, rosetta_path)
    rosetta_score(pdb_names[0], score_output_dir, rosetta_path)

    


    # 成功运行，输出结果文件路径
    print(f"Chai-Rosetta score files saved at {score_output_dir}.")
    # # 删除临时文件，即temp文件夹及其内容

    # shutil.rmtree(path["temp"])
    try:
        shutil.rmtree(path["temp"])
        print("Cleaned up temp directory.")
        os.remove("/home/users/hcdai/AI-peptide/Chai-1/chai-lab/RMSD_Time-test/run.py")
        print("Cleaned up run.py file.")
    except:
        pass  


    return score_output_dir

# 逐条从生成器中获取配体，并逐个与受体运行主函数，得到分数

## 将打分结果输入到csv文件中
def csv_add(path, score_output_dir, ligand_name, ligand_sequence, receptor_dict):
    '''将分数结果写入csv文件
    Args:
        csv_file_path: csv文件路径
        ligand_name: 配体名称
        ligand_sequence: 配体序列
        receptor_dict: 受体字典
        score_output_dir: 分数输出文件夹路径
    '''
    csv_file_path=path["output_csv"]
    pack_score_path=os.path.join(score_output_dir, "pack_input_score.sc")
    scores_score_path=os.path.join(score_output_dir, "scores.sc")
    receptor_name=list(receptor_dict.keys())[0]
    receptor_sequence=list(receptor_dict.values())[0]
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    # 创建目录（如果不存在）
    # csv_dir = os.path.dirname(csv_file_path)
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)

    # 初始化结果列表
    result_list = []

    # 读取pack_input_scores.sc文件，获取打分数据
    with open(pack_score_path, 'r') as sc_file:
            lines = sc_file.readlines()
            if len(lines) >= 3:
                headers = [f"pack_{header}" for header in lines[1].strip().split()[1:]]  # 第二行为表头，去掉第一个，并添加前缀
                data = lines[2].strip().split()     # 第三行为数据
                # 去掉第一个表头和第一个数据
                # headers = headers[1:]
                data = data[1:]
                # 创建字典存储打分数据
                pack_score_data = dict(zip(headers, data))
    # 获取scores.sc打分数据
    with open(scores_score_path, 'r') as sc_file:
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
    
    # 合并打分数据
    merged_scores_data = {**scores_data, **pack_score_data}

    result_list.append({
        'ligand_name': ligand_name,
        'ligand_sequence': ligand_sequence,
        'receptor_name': receptor_name,
        'receptor_sequence': receptor_sequence, 
        'time': time,
        **merged_scores_data  # 将打分数据作为额外字段添加
    })

    # 写入csv文件
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['ligand_name', 'ligand_sequence', 'receptor_name', 'receptor_sequence','time']
        # 添加打分数据的表头
        for key in result_list[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # writer.writeheader()
        for row in result_list:
            writer.writerow(row)
 
    print(f"Results have been written to {csv_file_path}")


# 实例化ligand_generator
ligand_generator_1 = ligand_generator()
csv_file_path=path["output_csv"]

# 生成csv文件供储存
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['ligand_name','ligand_sequence','receptor_name','receptor_sequence','time','scores_total_score','scores_dslf_fa13','scores_fa_atr','scores_fa_dun','scores_fa_elec','scores_fa_intra_rep','scores_fa_intra_sol_xover4','scores_fa_rep','scores_fa_sol','scores_hbond_bb_sc','scores_hbond_lr_bb','scores_hbond_sc','scores_hbond_sr_bb','scores_linear_chainbreak','scores_lk_ball_wtd','scores_omega','scores_overlap_chainbreak','scores_p_aa_pp','scores_pro_close','scores_rama_prepro','scores_ref','scores_yhh_planarity','scores_description','pack_total_score','pack_complex_normalized','pack_dG_cross','pack_dG_cross/dSASAx100','pack_dG_separated','pack_dG_separated/dSASAx100','pack_dSASA_hphobic','pack_dSASA_int','pack_dSASA_polar','pack_delta_unsatHbonds','pack_dslf_fa13','pack_fa_atr','pack_fa_dun','pack_fa_elec','pack_fa_intra_rep','pack_fa_intra_sol_xover4','pack_fa_rep','pack_fa_sol','pack_hbond_E_fraction','pack_hbond_bb_sc','pack_hbond_lr_bb','pack_hbond_sc','pack_hbond_sr_bb','pack_hbonds_int','pack_lk_ball_wtd','pack_nres_all','pack_nres_int','pack_omega','pack_p_aa_pp','pack_packstat','pack_per_residue_energy_int','pack_pro_close','pack_rama_prepro','pack_ref','pack_sc_value','pack_side1_normalized','pack_side1_score','pack_side2_normalized','pack_side2_score','pack_yhh_planarity','pack_description']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

# 运行主函数
while True:
    try:
        try:
            shutil.rmtree(path["temp"])
            logging.info("Cleaned up temp directory.")
            print("Cleaned up temp directory.")
            os.remove("/home/users/hcdai/AI-peptide/Chai-1/chai-lab/RMSD_Time-test/run.py")
            logging.info("Cleaned up run.py file.")
            print("Cleaned up run.py file.")
        except:
            logging.info("clean up failed.")
        ligand_name,ligand_sequence = ligand_generator_1.__next__()
        print(f"Processing {ligand_name}...")
        logging.info(f"Processing {ligand_name}...")
        for receptor_file, receptor_dict in receptors_dict.items():            
            logging.info(f"Processing {ligand_name} with {receptor_dict}...")
            score_output_dir = chai_rosetta_score(receptor_dict, 
                                                  ligand_name, 
                                                  ligand_sequence, 
                                                  receptor_chain_num = parameter["receptor_chain_num"], 
                                                  path = path)
            if score_output_dir is not None:
                csv_add(path, score_output_dir, ligand_name, ligand_sequence, receptor_dict)
            else:
                logging.warning(f"No score_output_dir for ligand {ligand_name} with receptor {receptor_dict}.")
            # csv_add(path, score_output_dir, ligand_name, ligand_sequence, receptor_dict)
            # 检查 score_output_dir 是否为 None
            
            # break
        print(f"Finished processing {ligand_name}.")
        logging.info(f"Finished processing {ligand_name}.")
        # break
    except StopIteration:
        try:
            shutil.rmtree(path["temp"])
            print("Cleaned up temp directory.")
            logging.info("Cleaned up temp directory.")
            os.remove("/home/users/hcdai/AI-peptide/Chai-1/chai-lab/RMSD_Time-test/run.py")
            logging.info("Cleaned up run.py file.")
            print("Cleaned up run.py file.")
            # 文件读写完毕，关闭日志
            logging.info("Finished processing all ligands.")
            logging.shutdown()
        except:
            logging.info("Finished processing all ligands.")
            logging.shutdown()
        break
    except:
        logging.exception("Error occurred while processing ligand.")
        logging.shutdown()
        print("Error occurred while processing ligand.")
        break







