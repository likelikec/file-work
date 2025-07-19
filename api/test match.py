import pandas as pd
from collections import defaultdict
import os


def read_first_file(file_path):
    """读取Excel文件并构建指标索引映射"""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {os.path.abspath(file_path)}")

    required_columns = ['class', 'metrics', 'yes']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"缺少必要列: {missing}")

    # 获取所有指标列信息（前3列为非指标）
    metrics_columns = df.columns[3:].tolist()
    metrics_indices = list(range(3, len(df.columns)))  # 实际列索引

    return df, metrics_columns, metrics_indices


def find_unique_combination(matches):
    """在匹配结果中寻找唯一的class-metrics组合"""
    if not matches:
        return None

    # 统计class-metrics组合出现次数
    counter = defaultdict(int)
    for match in matches:
        key = (match['class'], match['metrics'])
        counter[key] += 1

    # 寻找出现次数最多的组合
    max_count = max(counter.values())
    candidates = [k for k, v in counter.items() if v == max_count]

    if max_count == 1 or len(candidates) > 1:
        return None  # 无唯一组合

    # 返回唯一组合及其对应的yes值
    key = candidates[0]
    yes_values = [m['yes'] for m in matches if (m['class'], m['metrics']) == key]
    return {
        'class': key[0],
        'metrics': key[1],
        'yes': yes_values[0] if len(set(yes_values)) == 1 else 'Ambiguous'
    }


def process_second_file(df_train, metrics_indices, test_file):
    """处理测试文件并输出匹配结果"""
    try:
        df_test = pd.read_csv(test_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {os.path.abspath(test_file)}")

    # 转换并验证列名
    try:
        test_col_nums = [int(col) for col in df_test.columns]
        # 验证列号有效性
        if any(num < 1 or num > len(metrics_columns) for num in test_col_nums):
            raise ValueError("列号超出训练数据指标范围")

        # 转换为实际列索引（训练数据的第4列开始）
        test_indices = [3 + (num - 1) for num in test_col_nums]

        # 验证索引有效性
        if not all(3 <= idx < len(df_train.columns) for idx in test_indices):
            raise ValueError("存在无效的列索引")

        # 验证指标数量一致性
        if len(test_indices) != len(metrics_columns):
            print(f"警告: 测试文件指标数量({len(test_indices)})与训练数据({len(metrics_columns)})不一致")

    except ValueError as e:
        raise ValueError(f"处理测试文件列名时出错: {str(e)}")

    results = []
    for _, row in df_test.iterrows():
        # 提取测试行指标值
        try:
            query_values = [str(row[idx]) for idx in test_indices]
        except KeyError as e:
            raise ValueError(f"列索引不存在: {str(e)}")

        # 在训练数据中查找匹配行
        matches = []
        for _, train_row in df_train.iterrows():
            # 只比较指标列
            match = all(
                str(train_row[idx]) == qv
                for idx, qv in zip(test_indices, query_values)
            )
            if match:
                matches.append({
                    'class': train_row['class'],
                    'metrics': train_row['metrics'],
                    'yes': train_row['yes']
                })

        # 确定最终匹配结果
        result = find_unique_combination(matches)
        results.append({
            'input_values': query_values,
            'result': result
        })

    return results


if __name__ == "__main__":
    # 文件路径配置（使用原始字符串）
    excel_path = r"C:\Users\17958\Desktop\train_5.0.xlsx"
    csv_path = r"C:\Users\17958\Desktop\test_x.csv"

    # 执行处理流程
    try:
        print("正在加载训练数据...")
        df_train, metrics_columns, metrics_indices = read_first_file(excel_path)
        print(f"训练数据指标列: {metrics_columns}")
        print(f"总指标列数: {len(metrics_columns)}")

        print("正在处理测试数据...")
        results = process_second_file(df_train, metrics_indices, csv_path)

        print("\n匹配结果：")
        for idx, res in enumerate(results, 1):
            if res['result']:
                print(f"行 {idx}:")
                print(f"  Class: {res['result']['class']}")
                print(f"  Metrics: {res['result']['metrics']}")
                print(f"  Yes: {res['result']['yes']}")
                print(f"  输入指标: {res['input_values']}\n")
            else:
                print(f"行 {idx}: 未找到唯一匹配 → 输入指标: {res['input_values']}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")