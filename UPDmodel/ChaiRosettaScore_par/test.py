import csv

csv_file_path = '/home/users/hcdai/AI-peptide/ChaiRosettaScore/output_score/test.csv'
result_list = {'ligand_name': '1', 'ligand_sequence': 'MVKVLLFTGKTKEQG', 'receptor_name': '1', 'receptor_sequence': 'MVKVLLFTGKTKEQG', 'time': '0.00'}

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['ligand_name', 'ligand_sequence', 'receptor_name', 'receptor_sequence','time']
    # 添加打分数据的表头
        
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()