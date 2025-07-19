import pandas as pd
import re


def extract_denominator(value):
    """从类似'100% (10/10)'的字符串中提取分母"""
    match = re.search(r'\((\d+)/(\d+)\)', str(value))
    if match:
        return int(match.group(2))
    return None


def calculate_fraction(decimal, denominator):
    """根据小数和分母计算分数形式"""
    if pd.notna(decimal) and denominator is not None:
        numerator = round(decimal * denominator)
        return f"{numerator}/{denominator}"
    return None


def format_percentage_and_fraction(decimal, fraction):
    """将小数转换为百分比形式并与分数组合"""
    if pd.notna(decimal) and fraction is not None:
        percentage = f"{decimal * 100:.1f}%"  # 保留一位小数
        return f"{percentage} ({fraction})"
    return None


def main():
    file_path = r'C:\Users\17958\Desktop\tmp_processed.xlsx'  # Excel文件路径
    output_path = r'C:\Users\17958\Desktop\tmp_ok.xlsx'  # 输出文件路径

    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 提取lc-1和bc-1列的分母
        lc_1_denominators = df['lc-1'].apply(extract_denominator)
        bc_1_denominators = df['bc-1'].apply(extract_denominator)

        # 计算lc和bc的分数形式
        df['lc_fraction'] = df.apply(lambda row: calculate_fraction(row['lc'], lc_1_denominators[row.name]), axis=1)
        df['bc_fraction'] = df.apply(lambda row: calculate_fraction(row['bc'], bc_1_denominators[row.name]), axis=1)

        # 格式化结果为"百分比 (分数)"形式
        df['lc_formatted'] = df.apply(lambda row: format_percentage_and_fraction(row['lc'], row['lc_fraction']), axis=1)
        df['bc_formatted'] = df.apply(lambda row: format_percentage_and_fraction(row['bc'], row['bc_fraction']), axis=1)

        # 保存结果到新的Excel文件
        df.to_excel(output_path, index=False)
        print(f"已处理文件，输出到: {output_path}")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    main()