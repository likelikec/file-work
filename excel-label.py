import pandas as pd
import re
import numpy as np

# 定义文件路径
coverage_path = r"C:\Users\17958\Desktop\hits类覆盖率.xlsx"
train_path = r"C:\Users\17958\Desktop\train-defect4j修改版.xlsx"
train_1_path = r"C:\Users\17958\Desktop\train-1.xlsx"
train_2_path = r"C:\Users\17958\Desktop\train-2.xlsx"

# ### 第一步：处理“类覆盖率.xlsx”数据集
coverage_df = pd.read_excel(coverage_path, engine='openpyxl')
coverage_df.fillna('100%', inplace=True)
coverage_df.to_excel(coverage_path, index=False)

# ### 第二步：处理“train.xlsx”数据集并覆盖指定列
train_df = pd.read_excel(train_path, engine='openpyxl')
coverage_df.rename(columns={'lang-testart': 'testart', 'lang-mogul': 'mogul'}, inplace=True)
coverage_subset = coverage_df[['testart', 'CC', 'MC', 'LC', 'BC', 'mogul', 'CC-1', 'MC-1', 'LC-1', 'BC-1']]
train_df = train_df.merge(coverage_subset, on='testart', how='left', suffixes=('', '_coverage'))

for col in ['CC', 'MC', 'LC', 'BC', 'mogul', 'CC-1', 'MC-1', 'LC-1', 'BC-1']:
    train_df[col] = train_df[f'{col}_coverage'].combine_first(train_df[col])
    train_df.drop(columns=[f'{col}_coverage'], inplace=True)

train_df.to_excel(train_1_path, index=False)

# ### 第三步：对“train-1.xlsx”进行进一步处理
train_1_df = pd.read_excel(train_1_path, engine='openpyxl')

# 定义函数将百分比字符串转换为浮点数
def percent_to_float(x):
    if isinstance(x, str):
        match = re.search(r'(\d+\.?\d*)%', x)
        if match:
            percent_str = match.group(1)
            return float(percent_str) / 100
        else:
            return np.nan
    else:
        return x

# 将相关列转换为浮点数
for col in ['CC', 'MC', 'LC', 'BC', 'CC-1', 'MC-1', 'LC-1', 'BC-1']:
    train_1_df[col] = train_1_df[col].apply(percent_to_float)
    train_1_df[col] = pd.to_numeric(train_1_df[col], errors='coerce')

# 处理 NaN 值（这里填充为 0，可根据需求调整）
train_1_df.fillna(0, inplace=True)

# 处理“1适合LLM”列
if '1适合LLM' not in train_1_df.columns:
    train_1_df['1适合LLM'] = 2
else:
    train_1_df['1适合LLM'] = 2

# 定义条件并赋值
condition1 = (train_1_df['BC'] > train_1_df['BC-1']) & (train_1_df['LC'] >= train_1_df['LC-1'])
condition2 = (train_1_df['BC'] == train_1_df['BC-1']) & (train_1_df['LC'] >= train_1_df['LC-1'])
condition3 = (train_1_df['BC'] == train_1_df['BC-1']) & (train_1_df['LC'] < train_1_df['LC-1'])
condition4 = (train_1_df['BC'] < train_1_df['BC-1']) & (train_1_df['LC'] <= train_1_df['LC-1'])

train_1_df.loc[condition1 | condition2, '1适合LLM'] = 1
train_1_df.loc[condition3 | condition4, '1适合LLM'] = 0

# ### 第四步：保存处理后的数据
train_1_df.to_excel(train_2_path, index=False)

print("处理完成！新文件已保存为：", train_2_path)
