import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import PolynomialFeatures

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
file_path = r"C:\Users\17958\Desktop\train无2.xlsx"
df = pd.read_excel(file_path)
features = ['Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*', 'PDcy', 'PDpt',
           'OCavg', 'OCmax', 'WMC', 'CLOC', 'JLOC', 'LOC']
target = '1适合LLM'

# 异常值处理（示例处理OCmax）
df['OCmax'] = np.where(df['OCmax'] > df['OCmax'].quantile(0.95),
                      df['OCmax'].median(), df['OCmax'])

# 2. 非线性特征工程
# 创建多项式特征（捕捉倒U型关系）
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['LOC']])
df['LOC'] = poly_features[:,0]  # 保持原始特征
df['LOC_sq'] = poly_features[:,1]  # 二次项

# 3. 交互特征工程
interaction_features = ['OCavg', 'WMC']
df['OCavg_WMC'] = df['OCavg'] * df['WMC']

# 4. 可视化分析
# 创建画布
fig = plt.figure(figsize=(18, 15), dpi=100)
gs = fig.add_gridspec(3, 3)

# 4.1 倒U型关系可视化
ax1 = fig.add_subplot(gs[0, 0])

# 标准化LOC数据并扩展范围
df['LOC_scaled'] = (df['LOC'] - df['LOC'].mean()) / df['LOC'].std()
sns.regplot(x='LOC_scaled', y=target, data=df,
           order=2,
           scatter_kws={'alpha':0.3, 'color':'steelblue'},
           line_kws={'color':'red', 'linestyle':'--'},
           ci=95,  # 增加置信区间
           ax=ax1)
ax1.set_title('标准化LOC的二次趋势拟合')
ax1.set_xlabel('标准化后的代码行数')

# 4.2 交互效应热力图
ax2 = fig.add_subplot(gs[0, 1])
try:
    # 改用等宽分箱处理OCavg
    df['OCavg_bin'] = pd.cut(df['OCavg'], bins=3, duplicates='drop')
    pivot_table = df.pivot_table(values=target,
                                index='OCavg_bin',
                                columns=pd.qcut(df['WMC'], 3, duplicates='drop'),
                                aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2)
except ValueError as e:
    print(f"分箱失败，建议合并类别: {e}")
    # 备用方案：使用原始数值
    sns.scatterplot(x='OCavg', y='WMC', hue=target, data=df, ax=ax2)
    ax2.set_title('OCavg与WMC原始分布')

# 5. 阈值效应分析
# 5.1 决策树规则提取
ax3 = fig.add_subplot(gs[0, 2])
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(df[features + ['LOC_sq', 'OCavg_WMC']], df[target])
plot_tree(dt, feature_names=features + ['LOC_sq', 'OCavg_WMC'],
         class_names=['0','1'], filled=True, ax=ax3)
ax3.set_title('决策树阈值规则')

# 6. 综合模型分析
# 6.1 随机森林特征重要性
ax4 = fig.add_subplot(gs[1, :])
rf = RandomForestClassifier()
rf.fit(df[features + ['LOC_sq', 'OCavg_WMC']], df[target])
importances = pd.Series(rf.feature_importances_,
                       index=features + ['LOC_sq', 'OCavg_WMC'])
importances.sort_values().plot(kind='barh', ax=ax4)
ax4.set_title('特征重要性排名')

# 6.2 部分依赖图
ax5 = fig.add_subplot(gs[2, :])
PartialDependenceDisplay.from_estimator(
    rf, df[features + ['LOC_sq', 'OCavg_WMC']],
    features=['LOC', 'LOC_sq', 'OCavg_WMC'],
    grid_resolution=20, ax=ax5
)
ax5.set_title('关键特征边际效应')

# 保存分析报告
plt.tight_layout()
plt.savefig('advanced_analysis_report.png', dpi=300, bbox_inches='tight')
print("综合分析报告已保存为 advanced_analysis_report.png")

# 7. 数据重构建议输出
threshold_rules = []
for feature, threshold in zip(dt.tree_.feature, dt.tree_.threshold):
    if feature != -2:
        name = features + ['LOC_sq', 'OCavg_WMC'][feature]
        threshold_rules.append(f"{name} 阈值: {threshold:.2f}")

print("\n基于决策树的阈值规则建议：")
print("\n".join(threshold_rules[:3]))  # 显示前3条重要规则