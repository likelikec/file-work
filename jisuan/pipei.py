import pandas as pd


def process_excel_data(input_file_path, output_file_path):
    # 读取 Excel 文件
    df = pd.read_excel(input_file_path)

    # 获取 class 列的所有唯一值
    class_values = df['class'].unique()

    # 遍历 class-1 列中的每个值
    for index, row in df.iterrows():
        class_1_value = row['class-1']

        # 如果 class-1 的值不在 class 列中，则删除相关列的值
        if class_1_value not in class_values:
            df.loc[index, ['class-1', 'cc-1', 'mc-1', 'lc-1', 'bc-1', 'pro-1']] = None

    # 处理百分比格式，将"100%"转换为"100% (0/0)"
    for col in ['lc-1']:
        if col in df.columns:
            df[col] = df[col].apply(format_percentage)

    # 保存处理后的数据到新的 Excel 文件
    df.to_excel(output_file_path, index=False)
    print(f"处理完成，结果已保存到 {output_file_path}")


def format_percentage(value):
    """将纯百分比格式(如"100%")转换为带括号的格式(如"100% (0/0)")"""
    if isinstance(value, str) and value.endswith('%') and '(' not in value:
        try:
            # 提取百分比数值
            percent = float(value.strip('%'))
            if percent == 100:
                return "100% (0/0)"
            return value  # 非100%的值保持原样
        except ValueError:
            return value  # 转换失败时保持原样
    return value


# 指定文件路径
input_file_path = r"C:\Users\17958\Desktop\tmp.xlsx"
output_file_path = r"C:\Users\17958\Desktop\tmp_processed.xlsx"

# 处理数据
process_excel_data(input_file_path, output_file_path)