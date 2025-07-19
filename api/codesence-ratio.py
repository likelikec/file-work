import pandas as pd
import os


def calculate_ratios(input_file_path, output_file_path):
    """
    计算Excel文件中每个因变量对应的自变量占比

    参数:
    input_file_path (str): 输入Excel文件的路径
    output_file_path (str): 输出Excel文件的路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file_path)
        print(f"已读取文件，包含 {len(df)} 行数据")

        # 定义自变量列名
        independent_variables = [
            "String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface (UI)",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"
        ]

        # 检查所有自变量列是否存在
        missing_columns = [col for col in independent_variables if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件中缺少以下列: {', '.join(missing_columns)}")

        # 检查因变量列是否存在
        if "class_name" not in df.columns:
            raise ValueError("文件中缺少'class_name'列")

        # 遍历每一行数据
        for index, row in df.iterrows():
            # 计算当前行中'yes'的数量
            yes_count = sum(row[col] == 'yes' for col in independent_variables)

            # 计算每个自变量的占比
            for col in independent_variables:
                value = row[col]
                if value == 'yes':
                    ratio = yes_count / len(independent_variables)
                elif value == 'no':
                    ratio = 0
                else:
                    # 处理非'yes'或'no'的值
                    print(f"警告: 行 {index + 2} 的 {col} 列包含非'yes/no'值: {value}")
                    ratio = None

                # 更新数据框中的值
                df.at[index, col] = ratio

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存结果到新的Excel文件
        df.to_excel(output_file_path, index=False)
        print(f"已成功将结果保存到 {output_file_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return False

    return True


if __name__ == "__main__":
    # 文件路径配置
    INPUT_FILE_PATH = r"D:\project_py\file-work\api\codersence-all_filtered.xlsx"
    OUTPUT_FILE_PATH = r"D:\project_py\file-work\api\codersence-all_ratios.xlsx"

    # 执行计算和保存操作
    success = calculate_ratios(INPUT_FILE_PATH, OUTPUT_FILE_PATH)

    if success:
        print("操作已成功完成")
    else:
        print("操作未能成功完成，请检查错误信息")