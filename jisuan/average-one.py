import re
import pandas as pd
from collections import defaultdict


def parse_excel(file_path):
    """解析Excel文件内容，提取类名、BC和LC列的分子分母以及项目信息"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        data = []

        # 假设数据从第二行开始（跳过标题行）
        for index, row in df.iterrows():
            # 根据实际列名调整索引
            if len(row) < 6:
                continue

            class_name = row[0]
            pro = row[5]

            # 提取BC列的分子和分母
            bc_part = str(row[3])
            bc_match = re.search(r'\((\d+)/(\d+)\)', bc_part)
            if bc_match:
                bc_numerator = int(bc_match.group(1))
                bc_denominator = int(bc_match.group(2))
            else:
                bc_numerator = 0
                bc_denominator = 0

            # 提取LC列的分子和分母
            lc_part = str(row[4])
            lc_match = re.search(r'\((\d+)/(\d+)\)', lc_part)
            if lc_match:
                lc_numerator = int(lc_match.group(1))
                lc_denominator = int(lc_match.group(2))
            else:
                lc_numerator = 0
                lc_denominator = 0

            data.append({
                'class_name': class_name,
                'bc_numerator': bc_numerator,
                'bc_denominator': bc_denominator,
                'lc_numerator': lc_numerator,
                'lc_denominator': lc_denominator,
                'pro': pro
            })

        return data
    except Exception as e:
        print(f"解析Excel文件时出错: {e}")
        return []


def calculate_weighted_average(data):
    """按项目计算BC和LC的加权平均值"""
    project_bc = defaultdict(lambda: {'numerator': 0, 'denominator': 0})
    project_lc = defaultdict(lambda: {'numerator': 0, 'denominator': 0})

    for item in data:
        pro = item['pro']

        # 累加BC列的分子和分母
        project_bc[pro]['numerator'] += item['bc_numerator']
        project_bc[pro]['denominator'] += item['bc_denominator']

        # 累加LC列的分子和分母
        project_lc[pro]['numerator'] += item['lc_numerator']
        project_lc[pro]['denominator'] += item['lc_denominator']

    return project_bc, project_lc


def format_results(project_bc, project_lc):
    """格式化结果为指定输出格式"""
    results = []
    for pro in sorted(project_bc.keys()):
        bc_data = project_bc[pro]
        lc_data = project_lc[pro]

        # 计算BC百分比
        bc_percentage = (bc_data['numerator'] / bc_data['denominator'] * 100) if bc_data['denominator'] > 0 else 0
        # 计算LC百分比
        lc_percentage = (lc_data['numerator'] / lc_data['denominator'] * 100) if lc_data['denominator'] > 0 else 0

        results.append({
            'pro': pro,
            'bc': f"{bc_percentage:.1f}% ({bc_data['numerator']}/{bc_data['denominator']})",
            'lc': f"{lc_percentage:.1f}% ({lc_data['numerator']}/{lc_data['denominator']})"
        })
    return results


def main():
    """主函数，执行文件解析、计算和结果输出"""
    file_path = r"C:\Users\17958\Desktop\testart-rf.xlsx"

    try:
        # 检查文件是否存在
        import os
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在。")
            return

        data = parse_excel(file_path)
        if not data:
            print("错误: 未能从文件中提取有效数据。")
            return

        project_bc, project_lc = calculate_weighted_average(data)
        results = format_results(project_bc, project_lc)

        # 输出结果
        print("项目BC和LC的加权平均值:")
        for result in results:
            print(f"{result['pro']}: BC={result['bc']}, LC={result['lc']}")

    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    main()