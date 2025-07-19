import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\symtrain.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义自变量列名和因变量列名
independent_vars = ['B', 'COM_RAT', 'Cyclic', 'D',
                    'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
                    'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT',
                    'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
                    'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n',
                    'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
                    'Query', 'RFC', 'TODO', "String processing", "File operations", "Network communication",
                    "Database operations", "Mathematical calculation", "User Interface",
                    "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
                    "Concurrency and Multithreading", "Exception handling"]

dependent_var = '1适合LLM'

# 检查数据是否正确加载
print("数据预览：")
print(df.head())

# 检查列名是否存在
missing_cols = [col for col in independent_vars + [dependent_var] if col not in df.columns]
if missing_cols:
    print(f"以下列名在文件中未找到：{missing_cols}")
    raise ValueError("数据中缺少部分列，请检查！")

# 确保因变量是二元变量
if not df[dependent_var].isin([0, 1]).all():
    print(f"因变量 {dependent_var} 不是二元变量，请检查数据！")
    raise ValueError("因变量必须是 0 或 1 的二元变量")

# 1. 正态性检验
print("\n=== 正态性检验结果 ===")
normality_results = []

# 对每个变量进行正态性检验
for var in independent_vars:
    # 跳过非数值型变量
    if not pd.api.types.is_numeric_dtype(df[var]):
        print(f"变量 {var} 不是数值类型，跳过正态性检验")
        continue

    # 去除缺失值
    data = df[var].dropna()

    # 样本量大于5000时使用K-S检验，否则使用Shapiro-Wilk检验
    if len(data) > 5000:
        stat, p = stats.kstest(data, 'norm')
        test_method = 'K-S检验'
    else:
        stat, p = stats.shapiro(data)
        test_method = 'Shapiro-Wilk检验'

    # 偏度和峰度
    skew = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    normality_results.append({
        '变量': var,
        '检验方法': test_method,
        '统计量': stat,
        'P值': p,
        '是否正态(α=0.05)': '是' if p > 0.05 else '否',
        '偏度': skew,
        '峰度': kurtosis
    })

# 转换为DataFrame
normality_df = pd.DataFrame(normality_results)

# 输出结果
print("\n正态性检验结果汇总：")
print(normality_df)

# 保存结果到Excel
normality_df.to_excel("normality_test_results.xlsx", index=False)

# 2. 可视化正态性检验结果
# 创建一个大的图形
plt.figure(figsize=(20, 50))

# 随机选择10个变量进行可视化
np.random.seed(42)
sample_vars = np.random.choice(independent_vars, min(10, len(independent_vars)), replace=False)

# 对每个变量绘制直方图和QQ图
for i, var in enumerate(sample_vars):
    data = df[var].dropna()

    # 直方图
    plt.subplot(len(sample_vars), 2, 2 * i + 1)
    sns.histplot(data, kde=True)
    plt.axvline(data.mean(), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(data.median(), color='g', linestyle='dashed', linewidth=2)
    plt.title(f'{var} 的分布 (均值: {data.mean():.2f}, 中位数: {data.median():.2f})')
    plt.xlabel(var)
    plt.ylabel('频率')

    # QQ图
    plt.subplot(len(sample_vars), 2, 2 * i + 2)
    qqplot(data, line='s', ax=plt.gca())
    plt.title(f'{var} 的 QQ 图')

plt.tight_layout()
plt.savefig('normality_plots.png', dpi=300)
plt.show()

# 3. 因变量分布可视化（因变量是二元变量，单独处理）
plt.figure(figsize=(8, 6))
sns.countplot(x=dependent_var, data=df)
plt.title(f'因变量 {dependent_var} 的分布')
plt.xlabel(dependent_var)
plt.ylabel('数量')
plt.savefig('dependent_variable_distribution.png', dpi=300)
plt.show()

print("\n正态性检验结果已保存至 'normality_test_results.xlsx'")
print("正态性可视化图表已保存至 'normality_plots.png'")
print(f"因变量分布图表已保存至 'dependent_variable_distribution.png'")