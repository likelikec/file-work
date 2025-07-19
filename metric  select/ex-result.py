import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools
import numpy as np

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\train2.0_processed.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义自变量列名（52个）
independent_vars = [
    'WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'B', 'D', 'E', 'N', 'n', 'V',
    'CLOC', 'NCLOC', 'LOC', 'Cyclic', 'Dcy', 'Dcy*', 'DPT', 'DPT*', 'PDcy',
    'PDpt', 'Command', 'COM_RAT', 'CONS', 'MPC', 'NAAC', 'NAIC', 'NOAC',
    'NOIC', 'NOOC', 'NTP', 'Level', 'Level*', 'Inner', 'INNER', 'CSA', 'CSO',
    'CSOA', 'jf', 'JM', 'JLOC', 'OCavg', 'OCmax', 'OPavg', 'OSavg', 'OSmax',
    'Query', 'STAT', 'SUB', 'TCOM_RAT', 'TODO'
]

# 定义因变量列名
dependent_var = '1适合LLM'

# 检查数据是否正确加载
print("数据预览：")
print(df.head())

# 检查列名是否存在
missing_cols = [col for col in independent_vars + [dependent_var] if col not in df.columns]
if missing_cols:
    print(f"以下列名在文件中未找到：{missing_cols}")
    raise ValueError("数据中缺少部分列，请检查！")

# 检查因变量分布
print(f"\n因变量 {dependent_var} 的分布：")
print(df[dependent_var].value_counts(normalize=True))

# 1. 构建所有自变量的 2、3、4、5 组合交互项
interaction_terms = []
interaction_names = []

# 定义组合阶数
combinations_orders = [2, 3]

for order in combinations_orders:
    print(f"\n正在生成 {order} 阶组合...")
    for combo in itertools.combinations(independent_vars, order):
        # 交互项名称
        interaction_name = " × ".join(combo)
        interaction_names.append(interaction_name)
        # 计算交互项（所有变量相乘）
        interaction_value = df[combo[0]].copy()
        for var in combo[1:]:
            interaction_value *= df[var]
        df[interaction_name] = interaction_value
        interaction_terms.append(interaction_name)

print(f"\n总共生成了 {len(interaction_terms)} 个交互项。")

# 2. 计算每个交互项与因变量的 Spearman 相关性
results = []
for term in interaction_terms:
    # 去除 NaN 值
    valid_data = df[[term, dependent_var]].dropna()
    if len(valid_data) > 1:  # 确保有足够的数据点
        corr, p_value = spearmanr(valid_data[term], valid_data[dependent_var])
        results.append({
            'Interaction Term': term,
            'Spearman Correlation': corr,
            'P-value': p_value
        })
    else:
        results.append({
            'Interaction Term': term,
            'Spearman Correlation': None,
            'P-value': None
        })
        print(f"{term} 的数据不足，无法计算相关性")

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 输出结果
print("\n交互项的 Spearman 相关性结果：")
print(results_df)

# 3. 绘制交互项相关性条形图（仅显示前 20 个最显著的交互项）
# 按相关系数绝对值排序
results_df_sorted = results_df.dropna().sort_values(by='Spearman Correlation', key=abs, ascending=False)

# 选取前 20 个最显著的交互项
top_n = 20
if len(results_df_sorted) > top_n:
    results_df_sorted = results_df_sorted.head(top_n)

plt.figure(figsize=(15, 8))

# 创建从金黄色到深紫色的渐变色
colors = ['#FFD700', '#FFA500', '#FF4500', '#C71585', '#800080', '#4B0082']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 归一化相关系数到 [0, 1] 以映射颜色
min_corr, max_corr = results_df_sorted['Spearman Correlation'].min(), results_df_sorted['Spearman Correlation'].max()
norm = plt.Normalize(min_corr, max_corr)
colors_mapped = [cmap(norm(val)) for val in results_df_sorted['Spearman Correlation']]

# 绘制条形图
bars = plt.barh(results_df_sorted['Interaction Term'], results_df_sorted['Spearman Correlation'],
                color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)

# 设置标题和标签
plt.title(f'前 {top_n} 个最显著交互项与 {dependent_var} 的 Spearman 相关性', fontsize=16, pad=20)
plt.xlabel('Spearman 相关系数', fontsize=12)
plt.ylabel('交互项', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)

# 设置背景颜色为纯色
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')

plt.tight_layout()
plt.savefig('interaction_spearman_bar.png', dpi=300)
plt.show()

# 4. 保存结果到文件
results_df.to_excel("interaction_spearman_results.xlsx", index=False)
print("\n交互项相关性结果已保存至 'interaction_spearman_results.xlsx'")
print("交互项相关性条形图已保存至 'interaction_spearman_bar.png'")