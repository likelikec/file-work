import pandas as pd

# 定义函数将百分比字符串转换为float
def convert_to_float(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        # 分割可能存在的括号并取第一部分
        value_part = value.split('(')[0].strip()
        # 处理百分号
        if '%' in value_part:
            return float(value_part.rstrip('%')) / 100
        else:
            return float(value_part)
    else:
        # 如果已经是数值类型，直接转换
        return float(value)

# 定义函数根据条件填充"1适合LLM"列
def get_llm_value(row):
    BC = row['BC']
    BC1 = row['BC-1']
    LC = row['LC']
    LC1 = row['LC-1']

    # 检查是否有NaN值
    if pd.isna(BC) or pd.isna(BC1) or pd.isna(LC) or pd.isna(LC1):
        print(f"Warning: NaN found in row {row.name}: BC={BC}, BC-1={BC1}, LC={LC}, LC-1={LC1}")
        return 1  # 默认值

    # 条件判断
    if BC > BC1 and LC < LC1:
        return 2
    elif BC < BC1 and LC > LC1:
        return 2
    elif BC >= BC1 and LC >= LC1:
        return 1
    elif (BC == BC1 and LC < LC1) or (BC < BC1 and LC <= LC1):
        return 0
    else:
        return 3  # 其他情况

# 1. 读取Excel文件中的sheet1表格
file_path = r"C:\Users\17958\Desktop\覆盖率.xlsx"
df = pd.read_excel(file_path, sheet_name='sym', engine='openpyxl')

# 2. 处理所有相关列，转换为float
columns_to_convert = ['LC', 'BC', 'LC-1', 'BC-1']
for col in columns_to_convert:
    df[col] = df[col].apply(convert_to_float)

# 3. 对BC-1列的空值填充为1
df['BC-1'] = df['BC-1'].fillna(1)

# 4. 填充"1适合LLM"列
df['1适合LLM'] = df.apply(get_llm_value, axis=1)

# 5. 转换为整数类型
df['1适合LLM'] = df['1适合LLM'].astype(int)

# 6. 保存结果到新文件
output_file_path = r"C:\Users\17958\Desktop\symlabel.xlsx"
df.to_excel(output_file_path, index=False)

print("处理完成，结果已保存到:", output_file_path)