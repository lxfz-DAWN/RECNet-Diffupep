##part 1 fasta文件的处理
def parse_fasta(fasta_file):  
    """  
    解析FASTA文件，并将序列名称和对应的氨基酸序列（作为逗号分隔的字符串列表）保存为字典。  
      
    参数:  
    fasta_file (str): FASTA文件的路径。  
      
    返回:  
    dict: 字典，其中键是序列名称，值是氨基酸序列的字符串列表（由逗号分隔）。  
    """  
    sequences = {}  # 初始化一个空字典来存储序列  
    current_name = None  # 初始化当前序列的名称  
    current_sequence = []  # 初始化当前序列的字符列表  
  
# 假设这里有一个循环遍历fasta文件的每一行  
    with open(fasta_file, 'r') as file:  
        for line in file:  
            if line.startswith('>'):  # 判断是否为序列名称行    
                if current_name is not None:  # 如果之前已经收集到序列    
                    sequences[current_name] = current_sequence  # 保存当前序列为字符串  
                current_name = line[1:].strip()  # 更新当前序列名称，去除开头的'>'和可能的尾部空格    
                current_sequence = []  # 重置当前序列    
            elif line.strip():  # 忽略空行和仅包含空白的行    
                current_sequence.extend(line.strip())  # 将当前行的字符（去除首尾空白）添加到当前序列中    
    
    # 处理完所有行后，保存最后一个序列（如果存在）    
    if current_name is not None:    
        sequences[current_name] = current_sequence  # 确保最后一个序列也是字符串  
    
    # 现在sequences字典包含了所有的序列  
    return sequences
# 示例用法  
fasta_file = r'C:\Users\22301\Desktop\fdu课堂资料\大二-大三暑期\test1.txt'  # 假设你有一个名为example.fasta的文件  
sequences = parse_fasta(fasta_file)  
print(sequences)

