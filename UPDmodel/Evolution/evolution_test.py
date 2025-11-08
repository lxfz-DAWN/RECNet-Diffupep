from evolution import *

# 测试整个进化过程

import random

s = ['E', 'N', 'Q', 'K', 'L', 'I', 'A', 'N',
     'Q', 'F', 'N', 'S', 'A', 'I', 'G', 'K', 
     'I', 'Q', 'D', 'S', 'L', 'S', 'S', 'T', 
     'A', 'S', 'A', 'L', 'G', 'K', 'L', 'Q', 
     'D', 'V', 'V', 'N', 'Q', 'N', 'A', 'Q', 
     'A', 'L', 'N', 'T', 'L', 'V', 'K', 'Q', 
     'L', 'S', 'S', 'N', 'F', 'G', 'A', 'I', 
     'S', 'S', 'V', 'L', 'N', 'D', 'I', 'L', 
     'S', 'R']

path = "probability matrix test.csv"


seq_probability_matrix = read_probability_matrix(path)


# 测试evolution类
evo = evolution(s, probability_matrix = seq_probability_matrix)

j = 0
while j < 5:
    j += 1
    seq_name = f"seq{j}"
    seq = fasta_sequence(name=seq_name, sequence = evo.run_evolution())
    seq.parent = s
    seq.pair = random.randint(1, 100) # 随机生成的pair
    # seq.remove_dashes()
    seq.write_csv() # 写入csv文件
    seq.write_fasta() # 写入fasta文件
    
