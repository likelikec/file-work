import pandas as pd


def merge_excel_sheets(input_file, output_file):
    # 读取 Excel 文件
    try:
        xls = pd.ExcelFile(input_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'")
        return
    except Exception as e:
        print(f"错误：无法读取文件 '{input_file}': {e}")
        return

    # 创建一个空的 DataFrame 用于存储合并后的数据
    merged_data = pd.DataFrame()

    # 获取所有表名
    sheet_names = xls.sheet_names

    # 遍历每个工作表并合并数据
    for sheet_name in sheet_names:
        try:
            # 读取当前工作表的数据
            df = xls.parse(sheet_name)

            # 添加 project 列，值为当前工作表的名称
            df['project'] = sheet_name

            # 将当前工作表的数据添加到合并后的数据中
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        except Exception as e:
            print(f"警告：处理工作表 '{sheet_name}' 时出错: {e}")
            continue

    # 检查是否有数据被合并
    if merged_data.empty:
        print("错误：没有找到有效数据")
        return

    # 将合并后的数据写入新的 Excel 文件
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            merged_data.to_excel(writer, sheet_name='合并数据', index=False)
        print(f"成功合并数据并保存到 '{output_file}'")
    except Exception as e:
        print(f"错误：无法保存文件 '{output_file}': {e}")


if __name__ == "__main__":
    # 指定输入和输出文件路径
    input_file = r"D:\project_py\file-work\api\gemini-14-classification_results.xlsx"
    output_file = r"D:\project_py\file-work\api\gemini-14-classification_results_合并.xlsx"

    # 执行合并操作
    merge_excel_sheets(input_file, output_file)