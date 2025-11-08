import csv

def read_protein_data_by_position(csv_file_path):
    protein_data = {}

    with open(csv_file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # 跳过表头（第一行）
        headers = next(csvreader)
        
        # 遍历CSV文件的每一行（从第二行开始）
        for row in csvreader:
            pdb_id = row[0]  # 第1列（索引0）
            metric = row[3]  # 第4列（索引3）
            value = row[4]   # 第5列（索引4）
            
            # 将信息添加到字典中
            if pdb_id in protein_data:
                # 如果PDB_ID已经存在，则追加信息到列表中
                protein_data[pdb_id].append([metric, value])
            else:
                # 如果PDB_ID不存在，则创建新条目
                protein_data[pdb_id] = [[metric, value]]

    return protein_data

# # 示例用法
# csv_file_path = '/home/users/hcdai/AI-peptide/RunRosetta/data/GNN_cleaned_data.csv'
# protein_data_dict = read_protein_data_by_position(csv_file_path)
# print(protein_data_dict)

# 打印结果以验证

def get_csv_headers(csv_file_path):
    with open(csv_file_path, mode='r', newline='') as csvfile:
        # 使用DictReader来获取表头
        csvreader = csv.DictReader(csvfile)
        # DictReader的fieldnames属性包含了表头的列名列表
        headers = csvreader.fieldnames
        return headers
 
# # 示例用法
# csv_file_path = '/home/users/hcdai/AI-peptide/RunRosetta/model.csv'
# headers = get_csv_headers(csv_file_path)
# print(headers)




    
        
csv_file_path = '/home/users/hcdai/AI-peptide/RunRosetta/data/GNN_cleaned_data.csv'
data_dict = read_protein_data_by_position(csv_file_path)
output_csv_path = '/home/users/hcdai/AI-peptide/RunRosetta/model.csv'
model_csv_path = "/home/users/hcdai/AI-peptide/RunRosetta/TF_ori_withpath.csv"
fieldnames = get_csv_headers(model_csv_path)
print(fieldnames)
with open('/home/users/hcdai/AI-peptide/RunRosetta/model.csv', mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # 遍历字典并写入数据行
    for key, values in data_dict.items():
        # 使用name_mapping来获取ligand_name和receptor_name
        receptor_name = key
        ligand_name = key
        # print(values[0][1].lower())
        # print(key, values)
        row = {
            'ligand_name': ligand_name,
            'receptor_name': receptor_name,
            'kd': 'none' if not values or values[0][0].lower() != 'kd' else values[0][1],
            'ki': 'none' if not values or values[0][0].lower() != 'ki' else values[0][1],
            'IC50': 'none' if not values or values[0][0].lower() != 'ic50' else values[0][1],
        }
        # print(row)
        # 写入数据行
        writer.writerow(row)