import pandas as pd
import os


def process_excel_files(file_a_path, file_b_path, file_c_path, output_path):
    # 读取文件A和文件B的数据，假设第一行为表头
    df_a = pd.read_excel(file_a_path)
    df_b = pd.read_excel(file_b_path)
    df_c = pd.read_excel(file_c_path)

    # 创建结果DataFrame，结构与文件A和文件B类似
    result_columns = ['mio', 'CC', 'MC', 'BC', 'LC', 'pro']
    result_data = []

    # 处理文件C中的每一行
    for _, row_c in df_c.iterrows():
        class_name = row_c['mio']
        prediction_label = row_c['预测标签']

        # 根据预测标签选择对应的数据源
        if prediction_label == 0:
            source_df = df_a
        else:
            source_df = df_b

        # 在数据源中查找匹配的类名
        match = source_df[source_df['mio'] == class_name]

        # 如果找到匹配项且唯一，则添加到结果中
        if len(match) == 1:
            result_data.append(match.iloc[0][result_columns].tolist())
        # 如果未找到匹配项或不唯一，则添加默认值
        else:
            default_row = [class_name] + ['100% (0/0)'] * 4 + [row_c.get('project', 'unknown')]
            result_data.append(default_row)

    # 创建结果DataFrame并保存到新文件
    result_df = pd.DataFrame(result_data, columns=result_columns)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存到: {output_path}")


# 文件路径配置
file_a_path = r"C:\Users\17958\Desktop\evo-sym_筛选结果.xlsx"
file_b_path = r"C:\Users\17958\Desktop\sym_筛选结果.xlsx"
file_c_path = r"C:\Users\17958\Desktop\预测\XGBoost_sym_predictions.xlsx"
output_path = r"C:\Users\17958\Desktop\sym-xgb.xlsx"

# 执行处理
process_excel_files(file_a_path, file_b_path, file_c_path, output_path)