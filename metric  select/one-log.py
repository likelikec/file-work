import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\symtrain.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义自变量列名（52个）
# independent_vars = ['DPT*',  'Inner', 'LCOM',
#                       'PDpt',
#                     'TCOM_RAT',   'CLOC',  'CONS',  'INNER', 'jf', 'JLOC', 'Jm',  'MPC',
#                     'NTP',
#                      'TODO']

independent_vars = ['B', 'COM_RAT', 'Cyclic',
            'Dc+y', 'DIT', 'DP+T',  'Inner', 'LCOM', 'Level',
             'NOAC', 'NOC',  'PDcy', 'PDpt',
                'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
             'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l',
              'NOOC', 'NTP', 'OCavg', 'OPavg',
              'TODO', "String processing", "File operations",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]

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

# 确保因变量是二元变量
if not df[dependent_var].isin([0, 1]).all():
    print(f"因变量 {dependent_var} 不是二元变量，请检查数据！")
    raise ValueError("因变量必须是 0 或 1 的二元变量")

# 检查因变量分布
print(f"\n因变量 {dependent_var} 的分布：")
print(df[dependent_var].value_counts(normalize=True))

# 检查自变量中的 NaN 值
print("\n检查自变量中的 NaN 值：")
nan_counts = df[independent_vars].isna().sum()
print(nan_counts[nan_counts > 0])  # 只显示存在 NaN 的列

# 1. 标准化自变量
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[independent_vars] = scaler.fit_transform(df[independent_vars])

# 再次检查 NaN 值（标准化后可能仍存在 NaN）
print("\n标准化后，自变量中的 NaN 值：")
nan_counts_after = df_standardized[independent_vars].isna().sum()
print(nan_counts_after[nan_counts_after > 0])

# 2. 单变量逻辑回归（使用标准化后的数据）
results = []
for var in independent_vars:
    # 准备数据
    X = df_standardized[[var]].dropna()
    y = df_standardized.loc[X.index, dependent_var]  # 确保 X 和 y 的索引对齐
    X = sm.add_constant(X)  # 添加常数项（截距）

    # 拟合逻辑回归模型
    try:
        model = sm.Logit(y, X).fit(disp=0)  # disp=0 禁止打印迭代信息
        coef = model.params[var]  # 自变量的系数
        std_err = model.bse[var]  # 标准误差
        z_value = model.tvalues[var]  # Z 值
        p_value = model.pvalues[var]  # P 值
        # 计算 OR（比值比）
        odds_ratio = np.exp(coef)
        # 计算 95% 置信区间
        conf_int = model.conf_int(alpha=0.05).loc[var].values
        conf_int_or = np.exp(conf_int)  # 转换为 OR 的置信区间

        results.append({
            'Variable': var,
            'Coefficient': coef,
            'Std Error': std_err,
            'Z Value': z_value,
            'P-value': p_value,
            'Odds Ratio': odds_ratio,
            '95% CI Lower (OR)': conf_int_or[0],
            '95% CI Upper (OR)': conf_int_or[1]
        })
    except Exception as e:
        print(f"变量 {var} 的逻辑回归拟合失败：{e}")
        results.append({
            'Variable': var,
            'Coefficient': None,
            'Std Error': None,
            'Z Value': None,
            'P-value': None,
            'Odds Ratio': None,
            '95% CI Lower (OR)': None,
            '95% CI Upper (OR)': None
        })

# 转换为 DataFrame
results_df = pd.DataFrame(results)
results_df['Significance'] = results_df['P-value'].apply(
    lambda p:
    '***' if not pd.isna(p) and p < 0.001 else
    '**' if not pd.isna(p) and p < 0.01 else
    '*' if not pd.isna(p) and p < 0.05 else
    ''
)

# 合并置信区间为字符串（处理NaN值）
def format_ci(row):
    lower = row['95% CI Lower (OR)']
    upper = row['95% CI Upper (OR)']
    if pd.isna(lower) or pd.isna(upper):
        return 'N/A'
    else:
        return f"[{lower:.2f}, {upper:.2f}]"

results_df['95% CI (OR)'] = results_df.apply(format_ci, axis=1)

# 调整列顺序（新增两列在最右侧）
columns_order = [
    'Variable', 'Coefficient', 'Std Error', 'Z Value', 'P-value', 'Significance',
    'Odds Ratio', '95% CI (OR)', '95% CI Lower (OR)', '95% CI Upper (OR)'
]
results_df = results_df[columns_order]

# 输出结果（显示小数点后4位）
print("\n单变量逻辑回归结果（标准化后）：")
print(results_df.round(4))


# 输出结果
print("\n单变量逻辑回归结果（标准化后）：")
print(results_df.round(4))  # 保留4位小数便于查看

# 3. 绘制逻辑回归系数条形图（仅显示前 20 个最显著的变量）
# 按系数绝对值排序
results_df_sorted = results_df.dropna().sort_values(by='Coefficient', key=abs, ascending=False)

# 选取前 20 个最显著的变量
top_n = 63
if len(results_df_sorted) > top_n:
    results_df_sorted = results_df_sorted.head(top_n)

plt.figure(figsize=(15, 8))

# 创建从金黄色到深紫色的渐变色
colors = ['#FFD700', '#FFA500', '#FF4500', '#C71585', '#800080', '#4B0082']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 归一化系数到 [0, 1] 以映射颜色
min_coef, max_coef = results_df_sorted['Coefficient'].min(), results_df_sorted['Coefficient'].max()
norm = plt.Normalize(min_coef, max_coef)
colors_mapped = [cmap(norm(val)) for val in results_df_sorted['Coefficient']]

# 绘制条形图
bars = plt.barh(results_df_sorted['Variable'], results_df_sorted['Coefficient'],
                color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)

# 标注 P 值显著性并显示 OR 和置信区间
for i, bar in enumerate(bars):
    p_value = results_df_sorted.iloc[i]['P-value']
    or_value = results_df_sorted.iloc[i]['Odds Ratio']
    ci_lower = results_df_sorted.iloc[i]['95% CI Lower (OR)']
    ci_upper = results_df_sorted.iloc[i]['95% CI Upper (OR)']
    # 在条形图右侧标注 OR 和 95% CI
    label = f'OR={or_value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'
    if p_value < 0.05:
        label += ' *'
    plt.text(bar.get_width() + 0.02 if bar.get_width() >= 0 else bar.get_width() - 0.02,
             bar.get_y() + bar.get_height()/2, label,
             ha='left' if bar.get_width() >= 0 else 'right', va='center',
             color='black', fontsize=10)

# 设置标题和标签
plt.title(f'前 {top_n} 个自变量的标准化逻辑回归系数（与 {dependent_var}）\n(* 表示 P < 0.05)', fontsize=16, pad=20)
plt.xlabel('标准化逻辑回归系数', fontsize=12)
plt.ylabel('自变量', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)

# 设置背景颜色为纯色
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')

plt.tight_layout()
plt.savefig('logistic-sym.png', dpi=300)
plt.show()

# 4. 保存结果到文件
results_df.to_excel("logistic-sym.xlsx", index=False)
# 保存到Excel（保留完整数值精度）
results_df.to_excel("logistic-sym.xlsx", index=False)
print("标准化逻辑回归系数条形图已保存至 'logistic-sym.png'")