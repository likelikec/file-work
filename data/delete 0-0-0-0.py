import pandas as pd
from pathlib import Path


def process_and_save_excel(input_path):
    """ 处理并保存Excel文件的完整流程 """
    # 读取原始文件
    original_file = Path(input_path)
    df = pd.read_excel(original_file)

    # 验证列是否存在
    required_cols = ['LC', 'BC', 'LC-1', 'BC-1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺失必要列: {', '.join(missing_cols)}")

    # 创建删除条件
    delete_condition = (
            (df['LC'] == 0) &
            (df['BC'] == 0) &
            (df['LC-1'] == 0) &
            (df['BC-1'] == 0)
    )

    # 获取要删除的行信息
    deleted_rows = df[delete_condition].index
    excel_row_numbers = [i + 2 for i in deleted_rows]  # 转换为Excel行号

    # 生成新数据集
    cleaned_df = df[~delete_condition].reset_index(drop=True)

    # 生成输出路径
    output_path = original_file.parent / f"{original_file.stem}_处理后{original_file.suffix}"

    # 保存新文件
    cleaned_df.to_excel(output_path, index=False)

    return {
        "total_deleted": len(deleted_rows),
        "deleted_rows": excel_row_numbers,
        "original_size": len(df),
        "new_size": len(cleaned_df),
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    file_path = r"C:\Users\17958\Desktop\类覆盖率+指标train.xlsx"

    try:
        result = process_and_save_excel(file_path)

        print(f"[处理报告]")
        print(f"原始数据行数: {result['original_size']}")
        print(f"删除符合条件行数: {result['total_deleted']}")
        print(f"剩余数据行数: {result['new_size']}")
        print(f"删除的Excel行号列表: {result['deleted_rows']}")
        print(f"新文件已保存至: {result['output_path']}")

    except Exception as e:
        print(f"[处理失败] 错误信息: {str(e)}")
        print("建议检查: 1) 文件路径是否正确 2) Excel列名是否匹配")