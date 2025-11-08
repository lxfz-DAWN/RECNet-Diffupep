import pandas as pd
 
# 读取第一个 CSV 文件
file1 = '/home/users/hcdai/AI-peptide/RunRosetta/output_result_total.csv'
df1 = pd.read_csv(file1, header=None)  # 假设没有标题行
 
# 读取第二个 CSV 文件
file2 = '/home/users/hcdai/AI-peptide/RunRosetta/model_copy.csv'
df2 = pd.read_csv(file2, header=None)  # 假设没有标题行
 
# 创建一个空的 DataFrame 来存储合并后的结果
merged_df = df2.copy()
 
# 遍历第一个 CSV 文件的每一行
for index1, row1 in df1.iterrows():
    key1 = row1[0]  # 第一个 CSV 文件的第一列
    # 在第二个 CSV 文件中查找匹配的行
    matching_rows = df2[df2[0] == key1]
    if not matching_rows.empty:
        for index2, row2 in matching_rows.iterrows():
            # 假设只需要更新第一个匹配项
            merged_df.loc[index2, 1] = row1[1]  # 第一个 CSV 文件的第二列
            merged_df.loc[index2, 3] = row1[3]  # 第一个 CSV 文件的第四列
            # 如果只需要更新第一个匹配项，跳出内层循环
            break
 
# 将合并后的 DataFrame 写入新的 CSV 文件
merged_df.to_csv('merged_file.csv', index=False, header=False)
 
print("合并完成，结果已保存到 merged_file.csv")