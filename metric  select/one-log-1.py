import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\train_4.0.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义自变量列名（52个）
independent_vars = ['B', 'COM_RAT', 'Cyclic', 'D',
   'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
   'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
   'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
   'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
   'Query', 'RFC', 'TODO']

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

# 1. 使用 SimpleImputer 填充 NaN 值（使用均值填充）
imputer = SimpleImputer(strategy='mean')
df_imputed = df.copy()
df_imputed[independent_vars] = imputer.fit_transform(df[independent_vars])

# 再次检查 NaN 值是否已填充
print("\n填充 NaN 后，自变量中的 NaN 值：")
nan_counts_after = df_imputed[independent_vars].isna().sum()
print(nan_counts_after[nan_counts_after > 0])

# 2. 标准化自变量
scaler = StandardScaler()
df_standardized = df_imputed.copy()
df_standardized[independent_vars] = scaler.fit_transform(df_imputed[independent_vars])

# 3. 单变量逻辑回归（使用标准化后的数据）
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

        results.append({
            'Variable': var,
            'Coefficient': coef,
            'Std Error': std_err,
            'Z Value': z_value,
            'P-value': p_value
        })
    except Exception as e:
        print(f"变量 {var} 的逻辑回归拟合失败：{e}")
        results.append({
            'Variable': var,
            'Coefficient': None,
            'Std Error': None,
            'Z Value': None,
            'P-value': None
        })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 输出单变量逻辑回归结果
print("\n单变量逻辑回归结果（标准化后）：")
print(results_df)

# 4. 绘制标准化逻辑回归系数条形图（仅显示前 20 个最显著的变量）
# 按系数绝对值排序
results_df_sorted = results_df.dropna().sort_values(by='Coefficient', key=abs, ascending=False)

# 选取前 20 个最显著的变量
top_n = 20
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

# 标注 P 值显著性
for i, bar in enumerate(bars):
    p_value = results_df_sorted.iloc[i]['P-value']
    if p_value < 0.05:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, '*',
                 ha='center', va='center', color='black', fontsize=12, fontweight='bold')

# 设置标题和标签
plt.title(f'前 {top_n} 个自变量的标准化逻辑回归系数（与 {dependent_var}）\n(* 表示 P < 0.05)', fontsize=16, pad=20)
plt.xlabel('标准化逻辑回归系数', fontsize=12)
plt.ylabel('自变量', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)

# 设置背景颜色为纯色
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')

plt.tight_layout()
plt.savefig('standardized_logistic_regression_coefficients_bar.png', dpi=300)
plt.show()

# 5. 非线性模型（Random Forest 和 XGBoost，使用未标准化的数据，NaN 已填充）
# 准备数据
X = df_imputed[independent_vars]
y = df_imputed[dependent_var]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 5.1 Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测并评估
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest 分类报告：")
print(classification_report(y_test, rf_predictions))

# 特征重要性
rf_importances = pd.DataFrame({
    'Variable': independent_vars,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest 特征重要性：")
print(rf_importances)

# 5.2 XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 预测并评估
xgb_predictions = xgb_model.predict(X_test)
print("\nXGBoost 分类报告：")
print(classification_report(y_test, xgb_predictions))

# 特征重要性
xgb_importances = pd.DataFrame({
    'Variable': independent_vars,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nXGBoost 特征重要性：")
print(xgb_importances)

# 6. 绘制非线性模型的特征重要性条形图
# 6.1 Random Forest 特征重要性
rf_importances_top = rf_importances.head(top_n)
plt.figure(figsize=(15, 8))
bars = plt.barh(rf_importances_top['Variable'], rf_importances_top['Importance'],
                color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)
plt.title(f'前 {top_n} 个特征的 Random Forest 重要性（与 {dependent_var}）', fontsize=16, pad=20)
plt.xlabel('特征重要性', fontsize=12)
plt.ylabel('自变量', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig('random_forest_importances_bar.png', dpi=300)
plt.show()

# 6.2 XGBoost 特征重要性
xgb_importances_top = xgb_importances.head(top_n)
plt.figure(figsize=(15, 8))
bars = plt.barh(xgb_importances_top['Variable'], xgb_importances_top['Importance'],
                color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)
plt.title(f'前 {top_n} 个特征的 XGBoost 重要性（与 {dependent_var}）', fontsize=16, pad=20)
plt.xlabel('特征重要性', fontsize=12)
plt.ylabel('自变量', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.gca().set_facecolor('#FFFFFF')
plt.gcf().set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig('xgboost_importances_bar.png', dpi=300)
plt.show()

# 7. 保存结果到文件
results_df.to_excel("standardized_logistic_regression_results.xlsx", index=False)
rf_importances.to_excel("random_forest_importances.xlsx", index=False)
xgb_importances.to_excel("xgboost_importances.xlsx", index=False)

print("\n标准化逻辑回归结果已保存至 'standardized_logistic_regression_results.xlsx'")
print("Random Forest 特征重要性已保存至 'random_forest_importances.xlsx'")
print("XGBoost 特征重要性已保存至 'xgboost_importances.xlsx'")
print("标准化逻辑回归系数条形图已保存至 'standardized_logistic_regression_coefficients_bar.png'")
print("Random Forest 特征重要性条形图已保存至 'random_forest_importances_bar.png'")
print("XGBoost 特征重要性条形图已保存至 'xgboost_importances_bar.png'")