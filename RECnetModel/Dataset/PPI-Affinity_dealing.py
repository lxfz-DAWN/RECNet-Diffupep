# # %%

# import time
# import os
# import subprocess
# import shutil
# import datetime
# import csv
# import pandas as pd
# from Bio.PDB.PDBParser import PDBParser
# from Bio.PDB import Selection, PDBIO

# # %%
# # 读取pdb文件的索引信息
# index_csv_path = ''
# print(pdb_index.head())
# # ab = pdb_index.loc['1wej']["ligand_chain"].split(';')
# # print(ab)

# # %%
# # 定义待处理的pdb文件的生成器

# pdb_dir = config['input']
# pdb_file_list = os.listdir(pdb_dir)
# pdb_file_num = len(pdb_file_list)

# def pdb_generator():
#     "pdb文件生成器，用于input文件夹中读取pdb文件，仅限后缀为.pdb的文件"
#     for i in range(pdb_file_num):
#         pdb_file:str = os.path.join(pdb_dir, pdb_file_list[i])
#         if os.path.isfile(pdb_file) and pdb_file.endswith('.pdb'):
#             yield pdb_file
#         else:
#             print(f"{pdb_file} is not a valid pdb file.")




# # %%
# # 定义pdb文件处理的函数

# def pdb_parser(pdb_file: str):
#     """
#     解析pdb文件，返回序列等信息
#     """

#     # 获取pdb文件的名字
#     pdb_part = pdb_file.split('/')[-1].split('.')[:-1]
#     pdb_name = pdb_part[0]

#     # 检测pdb_name是否在pdb_index中
#     if pdb_name not in pdb_index.index:
#         print(f"{pdb_name} not in pdb_index")
#         return pdb_name


#     receptor_chain:list = pdb_index.loc[pdb_name]["receptor_chain"].replace(' ', '').split(',')
#     ligand_chain:list = pdb_index.loc[pdb_name]["ligand_chain"].replace(' ', '').split(',')


#     # 定义pdb对象（解析器）
#     parser = PDBParser(PERMISSIVE=1) # PERMISSIV 标签表示一些与PDB文件相关的问题会被忽略（注意某些原子和/或残基会丢失）。
#     structure = parser.get_structure(pdb_name, pdb_file)

#     # 从pdb对象中选取特定的链，并解析其序列


 
#     # 定义氨基酸三个字符映射方式
#     mapping = {'ALA': 'A', 
#                'ARG': 'R', 
#                'ASN': 'N', 
#                'ASP': 'D', 
#                'CYS': 'C', 
#                'GLN': 'Q', 
#                'GLU': 'E', 
#                'GLY': 'G', 
#                'HIS': 'H', 
#                'ILE': 'I', 
#                'LEU': 'L', 
#                'LYS': 'K', 
#                'MET': 'M', 
#                'PHE': 'F', 
#                'PRO': 'P', 
#                'SER': 'S', 
#                'THR': 'T', 
#                'TRP': 'W', 
#                'TYR': 'Y', 
#                'VAL': 'V',
#                'ACE': ''}
    
#     # 初始化序列
#     receptor_seq = {}
#     ligand_seq = {}

#     for r in receptor_chain:
#         for model in structure:
#             for chain in model:
#                 if chain.get_id() == r:
#                     receptor_seq.update({r:''.join([mapping.get(item,'') for item in [residue.get_resname().strip() for residue in chain]])})
#     for l in ligand_chain:
#         for model in structure:
#             for chain in model:
#                 if chain.get_id() == l:
#                     ligand_seq.update({l:''.join([mapping.get(item,'') for item in [residue.get_resname().strip() for residue in chain]])})
                    
                    

#     return pdb_name , ligand_seq , receptor_seq  


# def run_command(command):
#     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#     return result.stdout
    

# # %%
# def rosetta_score_changed(pdb_file_path, output_dir, receptor_chain, ligand_chain, rosetta_path:dict = rosetta_path, ):
#     import os
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)  
    
#     print("Running Rosetta scoring")
    

#     # 执行命令的函数
#     # def run_command(command):
#     #     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#     #     return result.stdout
    
#     rosetta_temp_path = os.path.join(config["temp"])
#     if not os.path.exists(rosetta_temp_path):
#         os.makedirs(rosetta_temp_path)
#     scorefile_path = os.path.join(output_dir, "scores.sc")
#     if os.path.exists(scorefile_path):
#         os.remove(scorefile_path)
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
#     ori_path = os.getcwd()
#     print(ori_path)
#     # 下策：改变工作路径
#     os.chdir(rosetta_temp_path)
#     print('os.dir done')
 
#     # 对输入文件进行打分
#     score_command = f"{rosetta_path['score_executable']} -s {pdb_file_path} -no_optH false -ignore_unrecognized_res -out:pdb"
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
#     print(output_dir)

#     # 从rosetta_temp_path中逐条读取文件相对路径，进行接口分析，分析结果保存至output_dir中下以ligand_name命名的文件夹中
#     for pdb_file in pdb_files:
#         scored_pdb_file = pdb_file.replace(".pdb", "_scored.pdb")
#         scored_pdb_path = os.path.join(rosetta_temp_path, scored_pdb_file)
        
#     # 进行接口分析，输出结果到指定的文件夹
#     analyze_command = f"{rosetta_path['InterfaceAnalyzer']} -s {scored_pdb_path} -fixedchains {''.join(receptor_chain)},{''.join(ligand_chain)} @{rosetta_path['pack_input_options']}"
#     run_command(analyze_command)

#     print(f"InterfaceAnalyzer successful for {scored_pdb_file}.")

#     # 复制temp文件夹内的所有文件文件到output_dir
#     for file in os.listdir(rosetta_temp_path):
#         # if file.endswith(".pdb") or file.endswith(".sc"):
#         if file.endswith(".sc"):
#             shutil.copy(os.path.join(rosetta_temp_path, file), os.path.join(output_dir, file))
        

#     # 下策：重新定义回旧工作路径    
#     os.chdir(ori_path)
#     print(os.getcwd())

#     print("Rosetta scoring successful.")

# # %%
# def csv_add(csv_file_path, score_output_dir, ligand_name, ligand_sequence, receptor_name, receptor_sequence):
#     '''将分数结果写入csv文件
#     Args:
#         csv_file_path: csv文件路径
#         ligand_name: 配体名称
#         ligand_sequence: 配体序列
#         receptor_name: 受体名称
#         receptor_sequence: 受体序列
#         score_output_dir: 分数输出文件夹路径
#     '''
#     pack_score_path=os.path.join(score_output_dir, "pack_input_score.sc")
#     scores_score_path=os.path.join(score_output_dir, "score.sc")
#     time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


#     # 创建目录（如果不存在）
#     # csv_dir = os.path.dirname(csv_file_path)
#     # if not os.path.exists(csv_dir):
#     #     os.makedirs(csv_dir)

#     # 初始化结果列表
#     result_list = []

#     # 读取pack_input_scores.sc文件，获取打分数据
#     with open(pack_score_path, 'r') as sc_file:
#             lines = sc_file.readlines()
#             if len(lines) >= 3:
#                 headers = [f"pack_{header}" for header in lines[1].strip().split()[1:]]  # 第二行为表头，去掉第一个，并添加前缀
#                 data = lines[2].strip().split()     # 第三行为数据
#                 # 去掉第一个表头和第一个数据
#                 # headers = headers[1:]
#                 data = data[1:]
#                 # 创建字典存储打分数据
#                 pack_score_data = dict(zip(headers, data))
#     # 获取scores.sc打分数据
#     with open(scores_score_path, 'r') as sc_file:
#         lines = sc_file.readlines()
#         if len(lines) >= 3:
#             headers = [f"scores_{header}" for header in lines[1].strip().split()[1:]]  # 第二行为表头，去掉第一个，并添加前缀
#            # 初始化一个字典来存储数据，键为表头，值为空列表
#         scores_data = {header: [] for header in headers}
#             # 从第三行开始读取数据（索引为2的行）
#         for line in lines[2:]:
#             # 分割每行的数据，并跳过第一个数据（索引为0的元素）
#             data = line.strip().split()[1:]
            
#             # 检查数据长度是否与表头长度一致
#             if len(data) != len(headers):
#                 raise ValueError(f"数据长度 {len(data)} 与表头长度 {len(headers)} 不一致")
            
#             # 将数据添加到对应的表头列表中
#             for i, header in enumerate(headers):
#                 scores_data[header].append(data[i])
    
#     # 合并打分数据
#     merged_scores_data = {**scores_data, **pack_score_data}

#     result_list.append({
#         'ligand_name': ligand_name,
#         'ligand_sequence': ligand_sequence,
#         'receptor_name': receptor_name,
#         'receptor_sequence': receptor_sequence, 
#         'time': time,
#         **merged_scores_data  # 将打分数据作为额外字段添加
#     })

#     # 写入csv文件
#     with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
#         fieldnames = ['ligand_name', 'ligand_sequence', 'receptor_name', 'receptor_sequence','time']
#         # 添加打分数据的表头
#         for key in result_list[0].keys():
#             if key not in fieldnames:
#                 fieldnames.append(key)
        
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         # writer.writeheader()
#         for row in result_list:
#             writer.writerow(row)
 
#     print(f"Results have been written to {csv_file_path}")

# # %%
# try:
#     shutil.rmtree(config["temp"])
#     print("Cleaned up temp directory.")
# except:
#     pass

# # 新建error_pdb.txt文件
# try:
#     with open(config["error_pdb"], 'w') as file:
#         pass
# except:
#     pass

# if not os.path.isfile(config["output_csv"]):
#     with open(config["output_csv"], mode='w', newline='', encoding='utf-8') as csv_file:
#             fieldnames = ['ligand_name','ligand_sequence','receptor_name','receptor_sequence','time','scores_total_score','scores_dslf_fa13','scores_fa_atr','scores_fa_dun','scores_fa_elec','scores_fa_intra_rep','scores_fa_intra_sol_xover4','scores_fa_rep','scores_fa_sol','scores_hbond_bb_sc','scores_hbond_lr_bb','scores_hbond_sc','scores_hbond_sr_bb','scores_linear_chainbreak','scores_lk_ball_wtd','scores_omega','scores_overlap_chainbreak','scores_p_aa_pp','scores_pro_close','scores_rama_prepro','scores_ref','scores_yhh_planarity','scores_description','pack_total_score','pack_complex_normalized','pack_dG_cross','pack_dG_cross/dSASAx100','pack_dG_separated','pack_dG_separated/dSASAx100','pack_dSASA_hphobic','pack_dSASA_int','pack_dSASA_polar','pack_delta_unsatHbonds','pack_dslf_fa13','pack_fa_atr','pack_fa_dun','pack_fa_elec','pack_fa_intra_rep','pack_fa_intra_sol_xover4','pack_fa_rep','pack_fa_sol','pack_hbond_E_fraction','pack_hbond_bb_sc','pack_hbond_lr_bb','pack_hbond_sc','pack_hbond_sr_bb','pack_hbonds_int','pack_lk_ball_wtd','pack_nres_all','pack_nres_int','pack_omega','pack_p_aa_pp','pack_packstat','pack_per_residue_energy_int','pack_pro_close','pack_rama_prepro','pack_ref','pack_sc_value','pack_side1_normalized','pack_side1_score','pack_side2_normalized','pack_side2_score','pack_yhh_planarity','pack_description']
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#             writer.writeheader()

# # %%
# # 实例化生成器

# for pdb_file_path in pdb_generator():
#     parser = tuple(pdb_parser(pdb_file_path))
#     print(pdb_file_path)
#     print(pdb_parser(pdb_file_path))  
#     print(parser)

#     if len(parser) != 3:
#         # 以追加模式将问题pdb写入error_pdb.txt文件
#         with open('/home/users/hcdai/AI-peptide/RunRosetta/error_pdb.txt', 'a') as f:
#             f.write("".join(list(parser)) + '\n')
#         continue
        
#     pdb_name, ligand_seq, receptor_seq = parser

#     output_path = os.path.join(config['output'], pdb_name)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     # 调用Rosetta的程序
#     # try:
#     #     time.sleep(1)
#     #     print("sleep ended")
#     rosetta_score_changed(
#         pdb_file_path,
#         output_path,
#         receptor_chain = pdb_index.loc[pdb_name]["receptor_chain"].split(';'),
#         ligand_chain = pdb_index.loc[pdb_name]["ligand_chain"].split(';'),
#         rosetta_path=config['rosetta'],
        
#     )
#     # except:
#     #     # 以追加模式将问题pdb写入error_pdb.txt文件
#     #     print("Rosetta scoring failed.")
#     #     with open('/home/users/hcdai/AI-peptide/RunRosetta/error_pdb.txt', 'a') as f:
#     #         f.write("".join(list(parser[0])) + "255"+ '\n' )
#     #     continue
            

#     # 写入csv文件
#     csv_add(
#         csv_file_path=config['output_csv'],
#         score_output_dir=output_path,
#         ligand_name = pdb_name,
#         ligand_sequence = ligand_seq,
#         receptor_name = pdb_name,  # !!!!!!!!!!!!!!!!!
#         receptor_sequence = receptor_seq,
#     )

#     try:
#         shutil.rmtree(config["temp"])
#         print("Cleaned up temp directory.")
#     except:
#         pass

# print('Done!')

# # %%


import os
import pandas as pd
from Bio.PDB.PDBParser import PDBParser

# 定义变量
workspace = "/your/workspace/path"  # 工作空间路径
pdb_dir = "/your/pdb/input/dir"  # PDB 文件输入目录
pdb_index_path = "/your/pdb/index.csv"  # PDB 索引 CSV 文件路径
output_csv_path = "/your/output.csv"  # 输出 CSV 文件路径

# 切换到工作空间
os.chdir(workspace)

# 读取 PDB 索引 CSV 文件
pdb_index = pd.read_csv(pdb_index_path, index_col=0)

# 定义氨基酸三个字符映射方式
mapping = {'ALA': 'A', 
           'ARG': 'R', 
           'ASN': 'N', 
           'ASP': 'D', 
           'CYS': 'C', 
           'GLN': 'Q', 
           'GLU': 'E', 
           'GLY': 'G', 
           'HIS': 'H', 
           'ILE': 'I', 
           'LEU': 'L', 
           'LYS': 'K', 
           'MET': 'M', 
           'PHE': 'F', 
           'PRO': 'P', 
           'SER': 'S', 
           'THR': 'T', 
           'TRP': 'W', 
           'TYR': 'Y', 
           'VAL': 'V',
           'ACE': ''}

# 定义解析 PDB 文件的函数
def pdb_parser(pdb_file: str, receptor_chain: list, ligand_chain: list):
    """
    解析 PDB 文件，返回序列等信息
    """
    # 获取 PDB 文件的名字
    pdb_part = pdb_file.split('/')[-1].split('.')[:-1]
    pdb_name = pdb_part[0]

    # 定义 PDB 对象（解析器）
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdb_name, pdb_file)

    # 初始化序列
    receptor_seq = {}
    ligand_seq = {}

    for r in receptor_chain:
        for model in structure:
            for chain in model:
                if chain.get_id() == r:
                    receptor_seq.update({r: ''.join([mapping.get(item, '') for item in [residue.get_resname().strip() for residue in chain]])})
    for l in ligand_chain:
        for model in structure:
            for chain in model:
                if chain.get_id() == l:
                    ligand_seq.update({l: ''.join([mapping.get(item, '') for item in [residue.get_resname().strip() for residue in chain]])})

    return pdb_name, ligand_seq, receptor_seq

# 初始化结果列表
results = []

# 遍历 PDB 索引 CSV 文件中的每一行
for index, row in pdb_index.iterrows():
    pdb_name = index
    pdb_file = os.path.join(pdb_dir, f"{pdb_name}.pdb")

    if not os.path.isfile(pdb_file):
        print(f"{pdb_file} is not a valid PDB file.")
        continue

    receptor_chain = row["receptor_chain"].replace(' ', '').split(',')
    ligand_chain = row["ligand_chain"].replace(' ', '').split(',')

    pdb_name, ligand_seq, receptor_seq = pdb_parser(pdb_file, receptor_chain, ligand_chain)

    # 将结果添加到列表中
    results.append({
        'pdb_name': pdb_name,
        'ligand_seq': ligand_seq,
        'receptor_seq': receptor_seq
    })

# 将结果保存到 CSV 文件中
result_df = pd.DataFrame(results)
result_df.to_csv(output_csv_path, index=False)

print('Done!')
