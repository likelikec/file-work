import pandas as pd
import re


def parse_percent(value):
    """解析百分比数据"""
    if isinstance(value, str):
        match = re.search(r'(\d+\.?\d*)%', value)
        return float(match.group(1)) / 100 if match else 0.0
    elif isinstance(value, (float, int)):
        return value
    else:
        return 0.0


# 读取Excel文件（确保文件存在且有标题行）
file_path = r'C:\Users\17958\Desktop\hits类覆盖率.xlsx'
df = pd.read_excel(
    file_path,
    sheet_name=0,
    header=0,  # 明确指定第一行为列名
    na_values=['', None]
)


D_COLUMN_NAME = 'LC'
E_COLUMN_NAME = 'BC'
I_COLUMN_NAME = 'LC-1'
J_COLUMN_NAME = 'BC-1'
K_COLUMN_NAME = '1适合LLM'


print("当前列名:", df.columns.tolist())

# 填充空值为0
df.fillna(0, inplace=True)

# 处理百分比列
for col in [D_COLUMN_NAME, E_COLUMN_NAME, I_COLUMN_NAME, J_COLUMN_NAME]:
    df[col] = df[col].apply(parse_percent)

# 计算K列逻辑
df[K_COLUMN_NAME] = 2  # 初始化列为0
for index, row in df.iterrows():
    d = row[D_COLUMN_NAME]
    e = row[E_COLUMN_NAME]
    i = row[I_COLUMN_NAME]
    j = row[J_COLUMN_NAME]

    if d >= i and e >= j:
        df.at[index, K_COLUMN_NAME] = 1
    elif d <= i and e <= j:
        df.at[index, K_COLUMN_NAME] = 0

# 保存结果
df.to_excel('output.xlsx', index=False)
print("处理完成！结果已保存至 output.xlsx")