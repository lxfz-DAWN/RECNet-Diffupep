# 导入相关代码库
import torch
import hashlib
import pandas as pd
import os
import csv
import subprocess
import shutil
import re

# 定义函数读取CSV文件并将其转换为概率矩阵

def read_probability_matrix(csv_file_path):
    """
    读取CSV文件并将其转换为概率矩阵
    
    参数:
        csv_file_path (str): CSV文件的路径
    
    返回:
        torch.Tensor: 归一化后的概率矩阵
    """
    try:
        # 读取CSV文件，忽略第一行（蛋白质名称），并假设第一列是索引（位点序数）
        df = pd.read_csv(csv_file_path, index_col=0)

        # 将DataFrame直接转换为PyTorch张量，指定为float32类型
        probability_matrix = torch.tensor(df.values, dtype=torch.float32)

        # 对每个维度进行归一化-------
        probability_matrix = probability_matrix / probability_matrix.sum(dim=1, keepdim=True)

        return probability_matrix
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
        return None

# fasta文件读取函数

def parse_fasta(fasta_file):  
    """  
    解析FASTA文件，并将序列名称和对应的氨基酸序列（作为逗号分隔的字符串列表）保存为字典。  
      
    参数:  
    fasta_file (str): FASTA文件的路径。  
      
    返回:  
    dict: 字典，其中键是序列名称，值是氨基酸序列的字符串列表（由逗号分隔）。  
    """  
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
      
    return sequences


class fasta_sequence:
    '''定义FASTA序列类，包含序列名和序列  
      
    Args:  
        name (str): 序列名  
        sequence (Union[str, list]): 序列，可以是字符串或字符列表，建议为列表  
      
    Attributes:  
        name (str): 序列名  
        sequence (list): 序列，以字符列表的形式存储  
        parent (Optional[str]): 继承自哪条序列的标识，默认为None  
        pair (Optional[int]): 序列的对接数据，默认为None  
    ''' 
    
    
    def __init__(self, name, sequence):
        self.name = name
        
        # 判断输入的是sequence是否为list类型，如果不是，则将其转换为list类型
        if isinstance(sequence, list):  
            if not all(isinstance(char, str) and len(char) == 1 for char in sequence):  
                raise ValueError('All elements in the sequence list must be single-character strings.')  
            self.sequence = sequence  
        elif isinstance(sequence, str):  
            self.sequence = [char for char in sequence]  
        else:  
            raise TypeError('The sequence should be a list of single-character strings or a string.')
            
        # 继承自哪条序列，默认为None
        self.parent = None
        # 序列的对接数据，默认为None
        self.pair = None      
        
        
    # 生成每一个序列对应的hash值
    def get_hash(self):  
        """使用SHA-256算法生成序列的哈希值"""
          
        # 将序列转换为字节串，因为hashlib需要字节串作为输入  
        sequence_bytes = ''.join(self.sequence).encode('utf-8')  
        # 创建sha256哈希对象  
        hash_object = hashlib.sha256(sequence_bytes)  
        # 获取十六进制格式的哈希值  
        hex_dig = hash_object.hexdigest()  
        return hex_dig
        
    def __str__(self):
        return f"{self.name}:{self.sequence}"
    
    
    # 创建一个write_fasta方法，将fasta_sequence的属性写入指定路径的csv文件中
    def write_csv(self, path = None):
        """将fasta序列写入csv文件"""
        
        
        # 如果没有指定路径，则默认写入data文件夹下FastaSequenceLibrary.csv文件        
        if path is None:
            path = os.path.join('data', 'FastaSequenceLibrary.csv')
            
            # 确保目录存在  
            directory = os.path.dirname(path)  
            if not os.path.exists(directory):  
                os.makedirs(directory)
            
        # 如果指定路径文件不是csv文件，则抛出异常
        elif not os.path.splitext(path)[1] == '.csv':
            raise ValueError('The specified path is not a csv file.')
        
        # 定义csv文件的字段名，并写入name, sequence, parent, pair四个字段  
        fieldnames = ['name', 'sequence', 'parent', 'pair']  
          
        # 检查文件是否存在以确定是否需要写入头部  
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0 
                
        # 采用追加模式打开文件
        with open(path, 'a', newline='', encoding='utf-8') as f:  
            writer = csv.DictWriter(f, fieldnames=fieldnames) 
            
            # 如果需要写入头部，则写入字段名  
            if write_header:  
                writer.writeheader()  
            
            # 写入fasta_sequence的各个属性  
            row = {  
                'name': self.name,  
                'sequence': ''.join(self.sequence),  
                'parent': ''.join(self.parent) if self.parent is not None else '',  
                'pair': self.pair if self.pair is not None else ''  
            }  
            writer.writerow(row) 
            
    # 创建一个write_fasta方法，将fasta_sequence的属性写入指定路径的fasta文件中
    def write_fasta(self, path = None):
        """将fasta序列写入fasta文件"""
        
        # 如果没有指定路径，则默认写入data文件夹下FastaSequenceLibrary.fasta文件        
        if path is None:
            path = os.path.join('data', 'FastaSequenceLibrary.fasta')
            
            # 确保目录存在  
            directory = os.path.dirname(path)  
            if not os.path.exists(directory):  
                os.makedirs(directory)
            
        # 如果指定路径文件不是fasta文件，则抛出异常
        elif not os.path.splitext(path)[1] == '.fasta':
            raise ValueError('The specified path is not a fasta file.')
        
        # 写入fasta文件
        with open(path, 'a', encoding='utf-8') as f:  
            f.write(f">{self.name}\n{''.join(self.sequence)}\n")  
    
    # 创建一个remove_dashes方法，将序列中的"-"字符移除
    def remove_dashes(self):  
        """移除序列中的"-"字符"""  
        self.sequence = [char for char in self.sequence if char != '-']

    #处理ene文件     
    def find_ene_scores(self):  
        """  
        遍历 self.pair_paths 中的每个文件路径，解析 ene 文件，并返回包含文件名和打分值的元组列表。  
  
        返回:  
        list: 列表，元素为元组，每个元组包含 ene 文件名和对应的打分结果（如果找到的话）。  
        如果没有找到足够的实数，则不添加该文件的元组到列表中。  
        """  
        pattern = r'-?\d+(\.\d+)?'  # 正则表达式来匹配实数  
        scores = []  # 用于存储文件名和对应的打分结果  
  
        # 遍历 self.pair_paths 中的每个文件路径  
        for path in self.PairPath:  
            real_numbers = []  # 重置为当前文件的实数列表
            ene_filename = os.path.basename(path)  
            try:  
                with open(path, 'r') as file:  
                    # 逐行读取文件  
                    for line in file:  
                        # 使用 findall 查找所有匹配正则表达式的部分  
                        matches = re.findall(pattern, line)  
                        # 将找到的匹配项添加到列表中  
                        real_numbers.extend(matches)  
  
                    # 如果找到了至少五个实数，则返回第五个作为打分值  
                    if len(real_numbers) >= 5:  
                        scores.append((ene_filename, float(real_numbers[4])))  
            except FileNotFoundError:  
                # 如果文件不存在，可以记录错误或忽略  
                print(f"Warning: File {path} not found.")  
            except Exception as e:  
                # 捕获其他可能的异常并打印错误信息  
                print(f"Error processing file {path}: {e}")  
  
        return scores  

    
    def analyze_scores(self):  
        """  
        分析 find_ene_scores 函数的输出，并基于给定的规则处理。  
  
        返回:  
        bool: 如果所有元组的第二个元素小于其对应的阈值，则返回 True；  
              如果存在至少一个元组的第二个元素不小于其阈值，则返回 False。  
        同时，更新 self.direction 为具有最大第二个元素的元组的第一个元素（文件路径）。  
        """  
        scores = self.find_ene_scores()  
        if not scores:  
            # 如果没有找到任何有效的打分，可能不需要更新 direction 或返回 True/False  
            print("找不到对应分数")  # 或者根据你的需求返回其他值  
  
        # 初始化 max_score 和 max_receptor  
        max_score = float('-inf')  
        max_receptor = None  
  
        # 遍历 scores 列表，检查每个元组的第二个元素  
        all_below_threshold = True  
        for ene_filename, score in scores:
            receptor=re.search(r"_(.*?)\.", ene_filename)  
            threshold = float(threshold_content[receptor]) - float(self.generation) * 2.5 #选取字典中的相应配体阈值  
            if score >= threshold:  
                all_below_threshold = False  
            if score > max_score:  
                max_score = score  
                max_receptor = receptor  
  
        # 更新 self.direction  
        self.direction = max_receptor  
  
        # 根据需求返回 True 或 False  
        return all_below_threshold

class FastaSequenceLibrary(fasta_sequence):  
    '''继承自fasta_sequence类，实现了序列库的功能，包括添加序列、检查序列是否存在、获取序列哈希值等功能
    
    Attributes:  
        sequences_hash: 字典，用于存储序列名和对应的哈希值
    
    '''
    
    def __init__(self):  
        # 使用字典来存储序列名和对应的哈希值  
        self.sequences_hash = {}  
  
    def add_sequence(self, name, sequence):  
        """添加新序列到库中，并计算其哈希值"""  
        hash_value = self.get_hash(sequence)  
        # 判断序列哈希值是否已存在于库中  
        if hash_value in self.sequences_hash.values():  
            return False          
  
        # 序列哈希值不存在，则添加到库中        
        self.sequences_hash[name] = hash_value  
        return True  

# 蛋白质序列进化，创建一个evolution类，输入序列和概率矩阵以及变异率，输出变异后的序列。

class evolution():
    '''定义进化类，包含序列进化的过程
    
    Attributes:  
    - sequence_to_be_evolved: 要进化的序列（字符串）。  
    - probability_matrix: 进化过程中氨基酸替换的概率矩阵。  
    - variability: 变异率。  
    
    Methods:  
    - mutate_probability_matrix: 使用变异率调整概率矩阵，并通过归一化保持每行概率和为1。  
    - evolve_sequence: 根据概率矩阵进化序列。  
    - run_mafft: 使用mafft进行序列比对。  
    - mutation_sequence: 按照比对结果进行序列更新。  
    - run_evolution: 运行进化过程。  
    '''
    
    def __init__(self, 
                 sequence_to_be_evolved, 
                 probability_matrix = None, 
                 variability = 0.05):  
        self.sequence_to_be_evolved = sequence_to_be_evolved
          
        if probability_matrix is None:  
            # 初始化为等概率（假设没有偏好）  
            self.probability_matrix = torch.ones((len(sequence_to_be_evolved), 20)) / 20  
        else:  
            self.probability_matrix = probability_matrix 
            
        self.variability = variability
        
        # 定义氨基酸索引到字符的映射  
        self.amino_acid_map = {  
            0: 'A', 
            1: 'C', 
            2: 'D', 
            3: 'E', 
            4: 'F', 
            5: 'G', 
            6: 'H', 
            7: 'I',  
            8: 'K', 
            9: 'L', 
            10: 'M', 
            11: 'N', 
            12: 'P', 
            13: 'Q', 
            14: 'R',  
            15: 'S', 
            16: 'T', 
            17: 'V', 
            18: 'W', 
            19: 'Y'  
        }   
    
    
    def mutate_probability_matrix(self):  
        '''使用变异率调整概率矩阵，并通过归一化保持每行概率和为1'''     
        
        # 生成与概率矩阵形状相同的正态分布噪声  
        noise = torch.randn_like(self.probability_matrix) * self.variability  
        # 更新概率矩阵并归一化  
        self.probability_matrix = torch.clamp(self.probability_matrix + noise, 0, 1)  
        self.probability_matrix = self.probability_matrix / self.probability_matrix.sum(dim=1, keepdim=True)  
  
    def evolve_sequence(self):  
        """根据概率矩阵进化序列"""  
        new_sequence = []  

        # 对每个位置随机选择新的氨基酸  
        for i in range(self.probability_matrix.shape[0]):  # 氨基酸个数与probability_matrix的行数相同
            
            # 从probability_matrix中获取当前位置的氨基酸替换概率  
            probs = self.probability_matrix[i]  
            
            # 使用torch.multinomial函数根据概率分布随机选择一个索引  
            # 这里probs是一个概率分布，torch.multinomial(probs, 1)会返回一个形状为(1,)的张量  
            # 其中的元素是根据probs中的概率随机选择的索引  
            # .item()用于将张量中的单个元素转换为Python标量  
            new_aa_index = torch.multinomial(probs, 1).item()  
            
            # 使用索引从映射字典中获取对应的氨基酸字符  
            new_aa = self.amino_acid_map[new_aa_index]  
            
            # 将新的氨基酸字符添加到新序列中  
            new_sequence.append(new_aa)        

        return new_sequence
    
    def run_mafft(self, new_sequence = [], sequence_to_be_evolved = [], output_file = None):
        '''  
        使用mafft进行序列比对。  
    
        参数:  
        - new_sequence: 一个包含新序列字符的列表（将被转换为字符串）。  
        - sequence_to_be_evolved: 表示要进化的序列（FASTA格式中的序列部分）。  
        - output_file: 比对结果将保存到的文件路径（默认为'alignment.fasta'）。  
        '''          
        
        print("开始进行序列比对")
        
        # 在当前路径中创建临时文件夹，缓存需要比对的序列文件        
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        if output_file is None:
            output_file = 'alignment.fasta'
        output_flie_path = os.path.join(temp_dir, output_file)
        
        # 保存new_sequence和sequence_to_be_evolved到临时文件夹
        new_sequence_str = ''.join(new_sequence)
        sequence_to_be_evolved_str = ''.join(sequence_to_be_evolved)
        # print(sequence_to_be_evolved)
        # print(sequence_to_be_evolved_str)
    
        with open(os.path.join(temp_dir, 'align.fasta'), 'w') as f:
            f.write(f'>new_sequence\n{new_sequence_str}\n>sequence_to_be_evolved\n{sequence_to_be_evolved_str}\n')
            
    
        # 使用mafft进行序列比对，需要指定了mafft的绝对路径
        mafft_path = "F:\mafft-win\mafft.bat" # 这里需要修改为自己的mafft路径
        mafft_command = [
            mafft_path,
            '--auto',
            # '--output', os.path.join(temp_dir, output_file), 
            os.path.join(temp_dir, 'align.fasta'), 
        ]
    
        try:  
            # 执行MAFFT命令，并等待其完成；这部分运行的命令可能需要按照linux系统进行重写
            with open(output_flie_path, 'w', encoding='utf-8') as O_F:  
                subprocess.run(mafft_command, check=True, text=True, stdout=O_F)
            

            # 读取比对结果
            fasta_dict = parse_fasta(output_flie_path)  
            
            # 打印比对结果  
            # print(fasta_dict)  
            
            
        except subprocess.CalledProcessError as e:
            fasta_dict = None
            print(f"Error executing MAFFT: {e}")  
            # 清理临时文件夹并重新抛出异常  
            shutil.rmtree(temp_dir, ignore_errors=True)  
            raise  # 重新抛出异常，以便调用者可以处理它
        
        except:
            fasta_dict = None
            raise  # 其他未知异常，也重新抛出
    
        finally:  
            # 清理临时文件（如果MAFFT成功执行并且文件已经被移动）  
            # 注意：如果文件已经被移动，这里不会删除任何东西  
            # 但如果MAFFT执行失败，并且没有抛出异常（比如被捕获了），这里将删除临时文件  
            # pass
            shutil.rmtree(temp_dir, ignore_errors=True) 
        # print(fasta_dict)
        return fasta_dict  
        
    def mutation_sequence(self, fasta_dict = None):
        """按照比对结果进行序列更新"""
        
        if fasta_dict is None:
            # 如果没有比对结果，则直接返回原序列
            return self.sequence_to_be_evolved
        else:
            mutation_sequence = fasta_dict['sequence_to_be_evolved']
        
        
        # 使用torch库选择突变位点的个数,突变个数服从正态分布
        mutation_num = abs(int(torch.normal(0, 25,(1,)) * self.variability)) + 1
        
        # 随机选择突变位点
        mutation_pos = torch.randint(0, len(fasta_dict['new_sequence']), (mutation_num,)).tolist()
        
        # 进行序列替代
        for i in mutation_pos:
            mutation_sequence[i] = fasta_dict['new_sequence'][i]
        
        
        return mutation_sequence
        
    def run_evolution(self):
        """运行进化过程"""
        
        # 进行变异
        self.mutate_probability_matrix()
        
        # 进行进化
        new_sequence = self.evolve_sequence()
        
        # 进行序列比对
        fasta_dict = self.run_mafft(new_sequence,self.sequence_to_be_evolved)
        
        
        # 按照比对结果进行序列更新
        mutation_sequence = self.mutation_sequence(fasta_dict)
        
        # 输出进化后的序列
        print(f"本次进化：{''.join(self.sequence_to_be_evolved)} -> {''.join(mutation_sequence)}")
        
        return mutation_sequence






