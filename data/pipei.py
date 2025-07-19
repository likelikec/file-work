import pandas as pd

# 读取文件（注意两个文件路径相同，请确认是否需要不同路径）
file1_path = r"C:\Users\17958\Desktop\jfreechart-label.xlsx"
file2_path = r"C:\Users\17958\Desktop\jfreechart指标.xlsx"

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# 处理class-1列：提取最后一个单词
df2['class-1_last_part'] = df2['class-1'].str.split('.').str[-1]

# 获取目标列位置
target_col = "1适合LLM"
insert_pos = df1.columns.get_loc(target_col) + 1

# 记录最大匹配次数
max_matches = 0
match_records = []

# 预扫描匹配次数
for _, row in df1.iterrows():
    matches = df2[df2['class-1_last_part'] == row['testart']]
    match_records.append(matches.drop(columns='class-1_last_part').to_dict('records'))
    max_matches = max(max_matches, len(matches))

# 生成动态列名
new_columns = []
for i in range(max_matches):
    for col in df2.columns.drop('class-1'):
        new_columns.append(f"匹配{i+1}_{col}")

# 插入空白列
for col in reversed(new_columns):
    df1.insert(insert_pos, col, None)

# 填充匹配数据
for idx, matches in enumerate(match_records):
    for match_num, match_data in enumerate(matches, 1):
        for col_name, value in match_data.items():
            df1.at[idx, f"匹配{match_num}_{col_name}"] = value

# 保存结果
output_path = r"C:\Users\17958\Desktop\横向拼接结果.xlsx"
df1.to_excel(output_path, index=False)

print(f"处理完成，最多匹配{max_matches}次，生成{len(new_columns)}个新列")