# 用于生成蛋白随机序列

import datetime
import random
import csv

print("This is the use.py file")

def generate_random_peptide(length):
    """
    生成随机蛋白序列
    
    :param length: 蛋白序列长度
    :return: 随机蛋白序列
    """
    
    # 定义氨基酸列表  
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY' 
    
    # 输入长度的有效性检查
    if length <= 0:
        raise ValueError("蛋白序列长度必须是正整数")
    
    # 生成随机序列，使用join提升效率
    peptide = ''.join(random.choice(amino_acids) for _ in range(length))
    
    return peptide

# 写入文件的函数
def write_peptide_to_file(peptide_name, peptide,length,time,path = None, filetype = 'csv'):
    
    if filetype == 'csv':
        if path is None:
            path = "./seq/random_peptide.csv"
            
        # 定义csv文件的字段名，并写入name, sequence, parent, pair四个字段  
        fieldnames = ['name','sequence', 'length', 'time']  
            
        # # 写入头部
        # with open(path, 'w', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        
        # 以追加模式打开csv文件，并写入数据
        with open(path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 写入蛋白序列到csv文件
            row = {  
                'name': peptide_name,
                'sequence': peptide,
                'length': length,
                'time': time
            }  
            writer.writerow(row) 
    elif filetype == 'fasta':
        if path is None:
            path = "./seq/random_peptide.fasta"
            
        # 写入fasta文件
        with open(path, 'a') as fastafile:
            fastafile.write('>' + peptide_name + '-' + str(time) +'\n')
            fastafile.write(peptide + '\n')
        
    return 1

# test

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

path = "/home/users/hcdai/AI-peptide/esm_test/NetRobustTest/seq/random_peptide.fasta"

for length in range(41, 60):
    for _ in range(5):
        peptide = generate_random_peptide(length)
        peptide_name = "random_peptide_" + str(length) + "_" + str(_)
        write_peptide_to_file(peptide_name, peptide, length, time, path, filetype='fasta')   