import pandas as pd

def replace_column_values():
    # 文件路径（注意使用原始字符串）
    file_path = r"C:\Users\17958\Desktop\类覆盖率+指标.xlsx"

    # 定义替换规则字典
    value_mapping = {
        0: "EVO",
        1: "LLM",
        2: "all"
    }

    try:
        # 读取Excel文件（自动检测引擎）
        df = pd.read_excel(file_path)

        # 验证目标列存在
        if "1适合LLM" not in df.columns:
            raise ValueError("列名'1适合LLM'不存在")

        # 执行替换操作（inplace方式更高效）
        df["1适合LLM"].replace(value_mapping, inplace=True)

        # 保存文件（保持原有格式）
        df.to_excel(file_path, index=False, engine="openpyxl")
        print("文件处理成功，已覆盖保存！")

    except Exception as e:
        print(f"处理失败，错误信息: {str(e)}")


if __name__ == "__main__":
    replace_column_values()