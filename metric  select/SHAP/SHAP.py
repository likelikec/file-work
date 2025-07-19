import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


# 读取Excel文件
file_path = r"C:\Users\17958\Desktop\train_4.0.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# 定义特征和目标列
features = ['B', 'COM_RAT', 'Cyclic', 'D', 'Dcy*', 'DIT', 'DPT*', 'E', 'Inner',
            'LCOM', 'Level', 'LOC', 'N', 'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax',
            'PDcy', 'PDpt', 'STAT', 'SUB', 'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC',
            'Command', 'CONS', 'CSA', 'CSO', 'CSOA', 'Dcy', 'DPT', 'INNER', 'jf',
            'JLOC', 'Jm', 'Level*', 'MPC', 'n', 'NAAC', 'NAIC', 'NOOC', 'NTP',
            'OCavg', 'OPavg', 'OSavg', 'OSmax', 'Query', 'RFC', 'TODO']
target = "1适合LLM"

# 数据预处理
X = df[features]
y = df[target]

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 可视化SHAP值
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
plt.savefig('global_importance.png', dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.savefig('shap_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== 数值分析模块 ====
global_importance = pd.DataFrame({
    'Feature': features,
    '|SHAP|': np.abs(shap_values).mean(axis=0),
    'SHAP_mean': shap_values.mean(axis=0)
}).sort_values('|SHAP|', ascending=False)

print("\n=== 全局特征影响力TOP20 ===")
print(global_importance.head(20).to_string(index=False))

# ==== 修正后的特征阈值分析 ====
top20_features = global_importance.head(20)['Feature'].tolist()


def auto_binning(feature_data, n_bins=4):
    """智能分箱函数"""
    try:
        return pd.qcut(feature_data, q=n_bins, duplicates='drop')
    except:
        return pd.cut(feature_data, bins=n_bins)


# ==== 修正后的特征阈值分析模块 ====
threshold_results = []
for feature in top20_features:
    safe_name = sanitize_filename(feature)

    # 生成分布直方图
    plt.figure(figsize=(10, 6))
    df[feature].hist(bins=20)
    plt.title(f'{feature} Distribution')
    plt.savefig(f'dist_{safe_name}.png', dpi=150)
    plt.close()

    # 分箱分析（修复列名问题）
    bins = auto_binning(X_test[feature], n_bins=4)
    bin_analysis = (
        X_test
        .groupby(bins)
        .apply(lambda x: pd.Series({
            '样本数': len(x),
            '正例率': y_test[x.index].mean(),
            '平均SHAP': shap_values[X_test.index.get_indexer(x.index)].mean()
        }))
        .rename_axis('分箱区间')  # 显式命名索引
        .reset_index()  # 转换为列
    )

    bin_analysis['特征'] = feature
    threshold_results.append(bin_analysis)

threshold_analysis = pd.concat(threshold_results)
print("\n=== 特征阈值分析 ===")
print(threshold_analysis.pivot(index='特征', columns='分箱区间', values=['样本数', '正例率']).to_string())

# ==== 特征交互分析 ====
top5_features = top20_features[:5]  # 选择前5特征生成C(5,2)=10种组合
interaction_analysis = []
for i in range(len(top5_features)):
    for j in range(i + 1, len(top5_features)):
        feat1, feat2 = top5_features[i], top5_features[j]
        # 计算交互效应
        interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)[:, features.index(feat1),
                             features.index(feat2)]
        interaction_analysis.append({
            '特征对': f"{feat1} × {feat2}",
            '平均交互效应': interaction_values.mean(),
            '|效应|': np.abs(interaction_values).mean()
        })

# 筛选前9强交互
interaction_df = pd.DataFrame(interaction_analysis).sort_values('|效应|', ascending=False).head(9)

# ==== 3x3矩阵可视化 ====
plt.figure(figsize=(18, 18), dpi=100)
colormaps = [
    plt.cm.cool,  # 第1列：蓝→品红
    plt.cm.Greens,  # 第2列：浅绿→深绿
    plt.cm.Oranges  # 第3列：黄→橙红
]

for idx, (_, row) in enumerate(interaction_df.iterrows()):
    ax = plt.subplot(3, 3, idx + 1)
    feat1, feat2 = row['特征对'].split(' × ')

    # 动态分配颜色映射
    col = idx % 3
    shap.dependence_plot(
        feat1, shap_values, X_test,
        interaction_index=feat2,
        ax=ax,
        cmap=colormaps[col],  # 列专属色系
        show=False,
        dot_size=14,  # 增大点尺寸
        alpha=0.7  # 添加透明度
    )
    ax.set_title(f"{feat1} ↔ {feat2}", fontsize=14, pad=10)
    ax.grid(alpha=0.3)
    ax.set_ylabel('')  # 清空默认Y轴标签

plt.tight_layout(pad=3.5)  # 增加子图间距
plt.savefig('interaction_3x3.png', bbox_inches='tight')
plt.close()

interaction_df = pd.DataFrame(interaction_analysis).sort_values('|效应|', ascending=False)
print("\n=== 关键特征交互效应 ===")
print(interaction_df.to_string(index=False))

# ==== 报告生成 ====
with open('shap_report.md', 'w') as f:
    f.write("# 详细SHAP分析报告\n\n")

    # 核心发现
    f.write("## 核心发现\n")
    f.write(
        f"- 主导特征 `{top20_features[0]}` 的绝对影响力是第二特征的{global_importance['|SHAP|'].iloc[0] / global_importance['|SHAP|'].iloc[1]:.1f}倍\n")
    f.write(f"- 最强特征交互：`{interaction_df.iloc[0]['特征对']}`（交互强度：{interaction_df.iloc[0]['|效应|']:.3f}）\n\n")

    # 特征分析
    f.write("## 特征详细分析\n")
    for feature in top20_features:
        safe_name = sanitize_filename(feature)
        f.write(f"### {feature}\n")
        f.write(f"![{feature}分布](dist_{safe_name}.png)\n")
        f.write(threshold_analysis[threshold_analysis['特征'] == feature].to_markdown(index=False))
        f.write("\n\n")

    # 交互分析
    f.write("## 关键交互效应\n")
    for _, row in interaction_df.iterrows():
        safe_pair = f"{sanitize_filename(row['特征对'].split(' × ')[0])}_{sanitize_filename(row['特征对'].split(' × ')[1])}"
        f.write(f"### {row['特征对']}\n")
        f.write(f"![交互效应图](interaction_{safe_pair}.png)\n")
        f.write(f"- 平均效应：{row['平均交互效应']:.3f}\n")
        f.write(f"- 绝对强度：{row['|效应|']:.3f}\n\n")

# 保存数据
global_importance.to_csv('feature_importance.csv', index=False)
pd.DataFrame(shap_values, columns=features).to_csv('shap_values.csv', index=False)

print("\n=== 分析完成 ===")
print("生成文件清单：")
print("- global_importance.png\t全局特征重要性")
print("- shap_distribution.png\t特征影响分布")
print("- dist_*.png\t\t特征分布直方图")
print("- interaction_*.png\t特征交互效应图")
print("- feature_importance.csv\t特征权重数据")
print("- shap_report.md\t完整分析报告")