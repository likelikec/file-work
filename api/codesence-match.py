import pandas as pd
import os


def match_and_filter_excel(symprompt_file, codersence_file, output_file=None):
    """
    读取两个Excel文件，将codersence文件中class_name列与symprompt文件中symprompt列进行匹配，
    筛选出匹配的行并按照symprompt列的顺序排序，最后保存为新的Excel文件。

    参数:
        symprompt_file (str): 包含symprompt列的Excel文件路径
        codersence_file (str): 包含class_name列的Excel文件路径
        output_file (str, optional): 输出文件路径。如果未提供，将在codersence文件同目录下生成。

    返回:
        str: 输出文件的路径
    """
    try:
        # 读取文件
        df_symprompt = pd.read_excel(symprompt_file)
        df_codersence = pd.read_excel(codersence_file)

        # 检查所需的列是否存在
        if 'symprompt' not in df_symprompt.columns:
            raise ValueError(f"文件 {symprompt_file} 中未找到 'symprompt' 列")

        if 'class_name' not in df_codersence.columns:
            raise ValueError(f"文件 {codersence_file} 中未找到 'class_name' 列")

        # 获取symprompt列的唯一值列表（保留顺序）
        symprompt_list = df_symprompt['symprompt'].dropna().tolist()
        symprompt_set = set(symprompt_list)

        # 创建class_name到其在symprompt中位置的映射
        position_map = {value: index for index, value in enumerate(symprompt_list)}

        # 筛选codersence文件中class_name列存在于symprompt值集合中的行
        filtered_df = df_codersence[df_codersence['class_name'].isin(symprompt_set)].copy()

        # 如果没有找到匹配项，输出警告
        if filtered_df.empty:
            print("警告: 未找到匹配的class_name值")
            return None

        # 按照symprompt中的顺序对筛选结果进行排序
        filtered_df['sort_key'] = filtered_df['class_name'].map(position_map)
        filtered_df.sort_values('sort_key', inplace=True)
        filtered_df.drop('sort_key', axis=1, inplace=True)

        # 如果未指定输出文件，生成默认文件名
        if not output_file:
            base_dir, filename = os.path.split(codersence_file)
            base_name, ext = os.path.splitext(filename)
            output_file = os.path.join(base_dir, f"{base_name}_filtered{ext}")

        # 保存筛选后的结果
        filtered_df.to_excel(output_file, index=False)
        print(f"已成功筛选并保存结果到 {output_file}")
        print(f"共筛选出 {len(filtered_df)} 行数据")

        return output_file

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    # 文件路径
    symprompt_file = r"C:\Users\17958\Desktop\匹配.xlsx"
    codersence_file = r"D:\project_py\file-work\api\codersence-all.xlsx"

    # 调用函数执行匹配和筛选
    output_file = match_and_filter_excel(symprompt_file, codersence_file)