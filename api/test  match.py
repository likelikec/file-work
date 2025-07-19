import pandas as pd

# 读取两个Excel文件
file1_path = r"C:\Users\17958\Desktop\all-res.xlsx"
file2_path = r"C:\Users\17958\Desktop\testart-test01.xlsx"

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# 获取文件2的testart列所有唯一值（转换为字符串类型）
testart_values = df2['mio'].astype(str).unique()

# 筛选文件1中class列值存在于testart_values中的行
# 同时将class列转换为字符串类型以确保类型匹配
filtered_df = df1[df1['class'].astype(str).isin(testart_values)]

# # 可选：重置索引（如果需要）
# filtered_df = filtered_df.reset_index(drop=True)

# 显示结果
print("筛选后的数据：")
print(filtered_df)

# 如果需要保存结果到新文件
output_path = r"C:\Users\17958\Desktop\geminiall_results.xlsx"
filtered_df.to_excel(output_path, index=False)
print(f"\n结果已保存到：{output_path}")