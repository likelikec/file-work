import pandas as pd
import os


def process_excel_files(file_a_path, file_b_path, output_path):
    # 检查文件是否存在
    for file_path in [file_a_path, file_b_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        # 读取文件a
        df_a = pd.read_excel(file_a_path)
        # 检查文件a是否包含必要的列
        if 'mio' not in df_a.columns:
            raise ValueError("文件a中未找到'mio'列")

        # 处理文件a中的空值，将其填充为"100% (0/0)"
        df_a = df_a.fillna("100% (0/0)")

        # 读取文件b
        df_b = pd.read_excel(file_b_path)
        # 检查文件b是否包含必要的列
        if 'mio' not in df_b.columns:
            raise ValueError("文件b中未找到'mio'列")

        # 获取文件b中所有唯一的类名
        b_class_names = set(df_b['mio'].unique())

        # 筛选文件a中在文件b中存在的类名对应的行
        filtered_df = df_a[df_a['mio'].isin(b_class_names)]

        # 保存结果到新文件
        filtered_df.to_excel(output_path, index=False)
        print(f"处理完成，结果已保存到: {output_path}")
        print(f"文件a原始行数: {len(df_a)}")
        print(f"筛选后保留的行数: {len(filtered_df)}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    # 文件路径配置
    file_a_path = r"C:\Users\17958\Desktop\evosuite.xlsx"
    file_b_path = r"C:\Users\17958\Desktop\预测\Random Forest_sym_predictions.xlsx"
    output_path = r"C:\Users\17958\Desktop\evo-sym_筛选结果.xlsx"

    # 执行处理
    process_excel_files(file_a_path, file_b_path, output_path)