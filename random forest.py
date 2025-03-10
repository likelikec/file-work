import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # 必须在其他 matplotlib 导入前设置
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    RocCurveDisplay  # 新增导入
)
# 读取数据
df = pd.read_excel(r"C:\Users\17958\Desktop\类覆盖率+指标.xlsx", engine='openpyxl')

# 预处理数据
# 删除因变量为2的行（第10列，索引9）
df = df[df['1适合LLM'] != 2]
df['1适合LLM'] = df['1适合LLM'].astype(int)

# 定义特征和标签
# 自变量列：Excel的O(14)到AC(28)列（pandas索引14到28）
features = ['Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*', 'PDcy',
           'PDpt', 'OCavg', 'OCmax', 'WMC', 'CLOC', 'JLOC', 'LOC', 'JF', 'JM']
X = df[features]
y = df['1适合LLM']
# 确保标签为二进制


# 划分训练集和测试集（分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 参数网格
param_grid = {
    'n_estimators': range( 50,200,20),
    'max_depth': [None, 3, 5,8,10],
    'min_samples_split': [3,5,8],
    'class_weight': ['balanced', None]
}

# 初始化随机森林
rf = RandomForestClassifier(random_state=42)

# 网格搜索（使用roc_auc作为评估指标）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 阈值优化（使用验证集）
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
best_model.fit(X_train_sub, y_train_sub)
y_proba_val = best_model.predict_proba(X_val)[:, 1]

# 寻找最佳阈值
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores[:-1])  # 排除最后一个元素
best_threshold = thresholds[best_threshold_idx]

# 应用最佳阈值到测试集
y_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= best_threshold).astype(int)

# 输出结果
print("最佳参数组合：")
print(grid_search.best_params_)
print("\n最佳分类阈值：", round(best_threshold, 3))
print("\n测试集分类报告：")
print(classification_report(y_test, y_pred_test))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_test))

# # 输出特征重要性
# importances = best_model.feature_importances_
# print("\n特征重要性：")
# for idx, imp in enumerate(importances):
#     print(f"特征 {X.columns[idx]}: {imp:.3f}")

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    name='Random Forest',
    color='darkorange'
)
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 添加对角线
plt.title('ROC Curve (Test Set)')
plt.grid(alpha=0.3)
plt.box(False)
plt.savefig('roc_curve.png', dpi=300)  # 保存图片
# plt.show()