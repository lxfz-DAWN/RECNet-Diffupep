import random
import pandas as pd

# 设置随机数种子以确保结果可重复
random.seed(42)

# 生成随机多肽序列
def generate_random_peptide(length):
    return ''.join(random.choices(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'], k=length))

# 生成数据
def generate_data(num_samples):
    sequences = [generate_random_peptide(random.randint(5, 15)) for _ in range(num_samples)]
    scores=[]
    for i in range(9):
        scores.append([random.uniform(0, 1) for _ in range(num_samples)])
    return pd.DataFrame({'sequence': sequences, 'score0': scores[0],'score1': scores[1],'score2': scores[2],
                         'score3': scores[3],'score4': scores[4],'score5': scores[5],
                         'score6': scores[6],'score7': scores[7],'score8': scores[8]})

# 生成数据
def generate_data(num_samples):
    sequences1 = [generate_random_peptide(random.randint(5, 15)) for _ in range(num_samples)]
    sequences2 = [generate_random_peptide(random.randint(5, 15)) for _ in range(num_samples)]
    scores=[random.uniform(0, 1) for _ in range(num_samples)]
    return pd.DataFrame({'sequence': sequences1, 'target_sequence': sequences2, 'score': scores})

# 生成训练集和验证集
train_data = generate_data(1000)
valid_data = generate_data(200)

# 保存为CSV文件
# train_data.to_csv("train_data.csv", index=False)
# valid_data.to_csv("valid_data.csv", index=False)
train_data.to_csv("train_data_plus.csv", index=False)
valid_data.to_csv("valid_data_plus.csv", index=False)

print("数据生成完毕并已保存为train_data.csv和valid_data.csv文件。")
