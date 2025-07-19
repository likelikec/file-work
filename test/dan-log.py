import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # 引入sklearn的逻辑回归（带正则化）
import random
from scipy import stats
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子确保结果可重现
np.random.seed(42)
random.seed(42)

# 文件路径
file_path = r"C:\Users\17958\Desktop\testarttrain.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义自变量列名
independent_vars =  ['B', 'COM_RAT', 'Cyclic',
                     'DIT', 'DP+T', 'Inner', 'LCOM','Dc+y',
                    'NOAC', 'NOC', 'PDcy', 'PDpt',
                    'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
                    'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level',
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


# 检查数据分布
def check_data_distribution(df, variables):
    """检查变量的数据分布（正态性检验）"""
    results = []
    for var in variables:
        data = df[var].dropna()
        if len(data) < 3:  # 至少需要3个样本点
            results.append({
                '变量': var,
                '统计量': np.nan,
                'p值': np.nan,
                '是否正态(α=0.05)': np.nan
            })
            continue

        stat, p = stats.shapiro(data)
        results.append({
            '变量': var,
            '统计量': stat,
            'p值': p,
            '是否正态(α=0.05)': p > 0.05
        })

    return pd.DataFrame(results)


# 检测完全分离（导致不收敛的常见原因）
def check_complete_separation(X, y):
    """检查是否存在完全分离（某个变量可完美预测因变量）"""
    # 对连续变量分组检查（离散化）
    X_vals = X.values.flatten()
    unique_vals = np.unique(X_vals)
    if len(unique_vals) > 10:  # 连续变量取分位数
        percentiles = np.percentile(X_vals, [25, 50, 75])
        thresholds = np.unique(percentiles)
    else:  # 离散变量直接用 unique 值
        thresholds = unique_vals

    for threshold in thresholds:
        mask = X_vals >= threshold
        y1 = y[mask]
        y0 = y[~mask]
        # 检查是否一组全为0另一组全为1
        if (len(y1) > 0 and len(y0) > 0) and (y1.sum() == len(y1) and y0.sum() == 0):
            return True
        if (len(y1) > 0 and len(y0) > 0) and (y1.sum() == 0 and y0.sum() == len(y0)):
            return True
    return False


# 使用自助法计算p值（适配多种模型）
def calculate_bootstrap_pvalue(X, y, var_index, n_bootstraps=1000):
    """使用自助法计算逻辑回归系数的p值（兼容不同拟合方法）"""
    # 先拟合基础模型确保收敛
    model = fit_logit_model(X, y)
    if model is None:
        return np.nan, np.nan, np.nan, np.nan
    original_coef = model.params[var_index]

    # 存储自助样本的系数
    bootstrap_coefs = []

    # 自助采样
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        try:
            model_boot = fit_logit_model(X_boot, y_boot)
            if model_boot is not None:
                bootstrap_coefs.append(model_boot.params[var_index])
        except:
            continue

    if len(bootstrap_coefs) < 100:
        return np.nan, original_coef, np.nan, np.nan

    bootstrap_coefs = np.array(bootstrap_coefs)
    p_value = 2 * min(
        np.mean(bootstrap_coefs >= abs(original_coef)),
        np.mean(bootstrap_coefs <= -abs(original_coef))
    )
    if p_value == 0:
        p_value = np.finfo(float).eps
    std_err = np.std(bootstrap_coefs)
    lower_ci = np.percentile(bootstrap_coefs, 2.5)
    upper_ci = np.percentile(bootstrap_coefs, 97.5)

    return p_value, original_coef, std_err, (lower_ci, upper_ci)


# 拟合逻辑回归模型（多种方法尝试解决不收敛）
def fit_logit_model(X, y):
    """尝试多种方法拟合逻辑回归，处理不收敛问题"""
    # 方法1：默认参数 + 增加迭代次数
    try:
        model = Logit(y, X).fit(
            disp=0,
            maxiter=1000,  # 增加迭代次数
            method='bfgs'  # 更稳定的优化算法
        )
        return model
    except:
        pass

    # 方法2：使用Firth修正（处理完全分离，需要statsmodels>=0.14.0）
    try:
        model = Logit(y, X).fit(
            disp=0,
            method='firth',  # Firth修正逻辑回归
            maxiter=1000
        )
        return model
    except:
        pass

    # 方法3：sklearn正则化逻辑回归（强正则化处理）
    try:
        # 提取自变量（移除常数项，sklearn会自动加）
        X_sk = X[:, 1:] if X.shape[1] > 1 else X
        model_sk = LogisticRegression(
            penalty='l2',  # L2正则化
            C=0.1,  # 强正则化（C越小正则化越强）
            solver='liblinear',  # 适合小样本
            max_iter=1000,
            random_state=42
        ).fit(X_sk, y)

        # 转换为类似statsmodels的结果格式（简化版）
        class SklearnModelWrapper:
            def __init__(self, coef, intercept):
                self.params = np.array([intercept, coef[0]]) if X.shape[1] > 1 else np.array([coef[0]])

        return SklearnModelWrapper(model_sk.coef_, model_sk.intercept_)
    except:
        pass

    # 所有方法都失败
    return None


# 2. 单变量逻辑回归（使用标准化后的数据，增强收敛处理）
results = []

# 选择p值计算方法：'traditional' 或 'bootstrap'
p_value_method = 'bootstrap'
n_bootstraps = 1000

# 先检查数据分布
print("\n=== 数据分布检查 ===")
distribution_results = check_data_distribution(df_standardized, independent_vars)
print("\n数据正态性检验结果（Shapiro-Wilk检验）：")
print(distribution_results)

normal_vars_ratio = distribution_results['是否正态(α=0.05)'].mean() * 100
print(f"\n正态分布变量比例：{normal_vars_ratio:.1f}%")

if p_value_method == 'bootstrap':
    print(f"\n将使用自助法({n_bootstraps}次采样)计算p值...")
else:
    print("\n将使用传统方法计算p值")

for var in independent_vars:
    # 准备数据
    X = df_standardized[[var]].dropna()
    y = df_standardized.loc[X.index, dependent_var]
    X = sm.add_constant(X)  # 添加常数项

    # 检查是否存在完全分离
    has_separation = check_complete_separation(X[[var]], y)
    if has_separation:
        print(f"警告：变量 {var} 存在完全分离，可能导致估计偏差")

    # 拟合模型（使用增强版拟合函数）
    model = fit_logit_model(X, y)
    if model is None:
        print(f"变量 {var} 所有拟合方法均失败")
        results.append({
            'Variable': var,
            'Coefficient': None,
            'Std Error': None,
            'Z Value': None,
            'P-value': None,
            'Odds Ratio': None,
            '95% CI Lower (OR)': None,
            '95% CI Upper (OR)': None,
            '拟合方法': '失败'
        })
        continue

    # 提取模型结果
    try:
        coef = model.params[var]
        # 处理不同模型的标准误（sklearn包装器没有bse，用自助法替代）
        if hasattr(model, 'bse'):
            std_err = model.bse[var]
            z_value = coef / std_err if std_err != 0 else np.nan
            p_value = model.pvalues[var] if p_value_method == 'traditional' else np.nan
        else:  # sklearn模型
            std_err = np.nan  # 简化处理，复杂可加自助法计算
            z_value = np.nan
            p_value = np.nan

        # 自助法计算p值（如果选择）
        if p_value_method == 'bootstrap':
            p_value, coef, std_err, conf_int = calculate_bootstrap_pvalue(
                X.values, y.values, X.columns.get_loc(var), n_bootstraps
            )
        else:
            conf_int = model.conf_int().loc[var].values

        # 处理p值显示
        if p_value == 0:
            p_value = np.finfo(float).eps

        # 计算OR和置信区间
        odds_ratio = np.exp(coef)
        conf_int_or = np.exp(conf_int) if conf_int is not None else (np.nan, np.nan)

        results.append({
            'Variable': var,
            'Coefficient': coef,
            'Std Error': std_err,
            'Z Value': z_value,
            'P-value': p_value,
            'Odds Ratio': odds_ratio,
            '95% CI Lower (OR)': conf_int_or[0],
            '95% CI Upper (OR)': conf_int_or[1],
            '拟合方法': 'Firth' if 'firth' in model.method else '传统' if model.method == 'bfgs' else '正则化'
        })
    except Exception as e:
        print(f"变量 {var} 结果提取失败：{e}")
        results.append({
            'Variable': var,
            'Coefficient': None,
            'Std Error': None,
            'Z Value': None,
            'P-value': None,
            'Odds Ratio': None,
            '95% CI Lower (OR)': None,
            '95% CI Upper (OR)': None,
            '拟合方法': '部分失败'
        })

# 转换为DataFrame
results_df = pd.DataFrame(results)


# 计算显著性标记
def get_significance(p_value):
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


results_df['Significance'] = results_df['P-value'].apply(get_significance)


# 合并置信区间为字符串
def format_ci(row):
    lower = row['95% CI Lower (OR)']
    upper = row['95% CI Upper (OR)']
    if pd.isna(lower) or pd.isna(upper):
        return 'N/A'
    else:
        return f"[{lower:.2f}, {upper:.2f}]"


results_df['95% CI (OR)'] = results_df.apply(format_ci, axis=1)

# 调整列顺序
columns_order = [
    'Variable', 'Coefficient', 'Std Error', 'Z Value', 'P-value', 'Significance',
    'Odds Ratio', '95% CI (OR)', '拟合方法'
]
results_df = results_df[columns_order]

# 输出结果
print(f"\n单变量逻辑回归结果（标准化后，p值计算方法：{p_value_method}）：")
with pd.option_context(
        'display.float_format',
        lambda x: f'{x:.10e}' if 'P-value' in str(x) else f'{x:.4f}'
):
    print(results_df)

# 3. 绘制逻辑回归系数条形图
results_df_sorted = results_df.dropna(subset=['Coefficient']).sort_values(by='Coefficient', key=abs, ascending=False)
top_n = min(63, len(results_df_sorted))
if len(results_df_sorted) > top_n:
    results_df_sorted = results_df_sorted.head(top_n)

plt.figure(figsize=(15, 10))

# 自定义渐变色
colors = ['#FFD700', '#FFA500', '#FF4500', '#C71585', '#800080', '#4B0082']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 映射颜色
min_coef, max_coef = results_df_sorted['Coefficient'].min(), results_df_sorted['Coefficient'].max()
norm = plt.Normalize(min_coef, max_coef)
colors_mapped = [cmap(norm(val)) for val in results_df_sorted['Coefficient']]

# 绘制条形图
bars = plt.barh(results_df_sorted['Variable'], results_df_sorted['Coefficient'],
                color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)

# 标注显著性和OR值
for i, bar in enumerate(bars):
    p_value = results_df_sorted.iloc[i]['P-value']
    or_value = results_df_sorted.iloc[i]['Odds Ratio']
    ci_text = results_df_sorted.iloc[i]['95% CI (OR)']
    sig = results_df_sorted.iloc[i]['Significance']

    p_formatted = f"{p_value:.2e}" if not pd.isna(p_value) else "N/A"
    label = f'OR={or_value:.2f} {ci_text}\nP={p_formatted} {sig}'

    plt.text(bar.get_width() + 0.02 if bar.get_width() >= 0 else bar.get_width() - 0.02,
             bar.get_y() + bar.get_height() / 2, label,
             ha='left' if bar.get_width() >= 0 else 'right', va='center',
             color='black', fontsize=7)

plt.title(f'前 {top_n} 个自变量的标准化逻辑回归系数（与 {dependent_var}）\n(包含收敛优化处理)', fontsize=16, pad=20)
plt.xlabel('标准化逻辑回归系数', fontsize=12)
plt.ylabel('自变量', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig(f'logistic-testart_{p_value_method}_converged.png', dpi=300)
plt.show()

# 4. 保存结果
results_df.to_excel(f"logistic-testart_{p_value_method}_converged.xlsx", index=False)
print(f"结果已保存至 'logistic-testart_{p_value_method}_converged.xlsx' 和对应的图片文件")