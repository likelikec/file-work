import pandas as pd
from scipy.stats import pointbiserialr

# 读取Excel文件（注意路径需存在且列名正确）
file_path = r"C:\Users\17958\Desktop\train-1.0.xlsx"
df = pd.read_excel(file_path)

# 定义目标列和二分类列
target_columns = ["Cyclic", "Dcy", "Dcy*", "Dpt", "Dpt*", "PDcy", "PDpt",
                  "OCavg", "OCmax", "WMC", "CLOC", "JLOC", "LOC", "JF", "JM"]
binary_column = "1适合LLM"  # 确保列名与实际文件一致

# 检查二分类列是否合法
if binary_column not in df.columns:
    raise ValueError(f"列 '{binary_column}' 不存在！")

# 确保二分类列是数值型（0/1或类似编码）
if df[binary_column].nunique() != 2:
    raise ValueError(f"列 '{binary_column}' 不是二分类变量！")

# 初始化结果存储
results = {
    "指标": [],
    "点二列相关系数": [],
    "p值": []
}

# 遍历每个指标列并计算点二列相关
for col in target_columns:
    if col not in df.columns:
        print(f"警告: 列 '{col}' 不存在，跳过")
        continue

    # 删除缺失值（可选：根据需求调整处理方式）
    data = df[[col, binary_column]].dropna()
    x = data[col].astype(float)  # 确保为连续变量
    y = data[binary_column].astype(int)  # 确保为二分类

    # 计算点二列相关系数
    corr, p_value = pointbiserialr(x, y)

    # 保存结果
    results["指标"].append(col)
    results["点二列相关系数"].append(corr)
    results["p值"].append(p_value)

# 转换为DataFrame并展示结果
results_df = pd.DataFrame(results)
print(results_df)

# 可选：保存结果到CSV
results_df.to_csv("点二列相关分析结果.csv", index=False, encoding="utf_8_sig")