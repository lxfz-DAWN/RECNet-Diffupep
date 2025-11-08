import pandas as pd
import numpy as np

# python /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/testandlearn.py
# 读取CSV文件
df = pd.read_csv('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/trying/20250217_loss_too_big/dataset/generated_EK_withpath_train.csv')

# 提取数值列（自动排除非数值列如字符串、日期等）
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 对每个数值列进行Z-score标准化
for col in numeric_cols:
    # 计算均值和标准差（自动忽略NaN）
    col_mean = df[col].mean(skipna=True)
    col_std = df[col].std(skipna=True)
    
    # 处理标准差为0的情况
    if col_std != 0:
        df[col] = (df[col] - col_mean) / col_std
    else:
        df[col] = 0  # 若标准差为0，则所有值为0

# 保持原有顺序并保存到新文件
df.to_csv('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/trying/20250217_loss_too_big/dataset/train_zscore.csv', index=False)