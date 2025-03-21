import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, classification_report, roc_curve, auc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义列名常量
FEATURE_COLS = [
    'Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*',
    'PDcy', 'PDpt', 'OCavg', 'OCmax', 'WMC',
    'CLOC', 'JLOC', 'LOC', 'JF', 'JM'
]
TARGET_COL = '1适合LLM'  # 假设第10列的列名是'Label'

# 读取数据并预处理
data = pd.read_excel(r"C:\Users\17958\Desktop\类覆盖率+指标.xlsx")

# 删除标签列中值为2的行
data = data[data[TARGET_COL] != 2]

# 定义特征和标签
X = data[FEATURE_COLS]  # 使用列名选择特征
y = data[TARGET_COL].astype(int)  # 使用列名选择标签

# 检查数据维度
print(f"特征矩阵形状: {X.shape}, 标签形状: {y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 处理类别不平衡（保持原始逻辑）
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])



# 定义XGBoost模型
model = xgb.XGBClassifier(objective='binary:logistic',
                          eval_metric='auc',
                          scale_pos_weight=scale_pos_weight)

# 参数网格
param_grid = {
    'max_depth': [3, 5, 7,9,10],
    'learning_rate': [0.01, 0.1, 0.12,0.15,0.18,0.2,0.21,0.24,0.26],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'n_estimators': [50, 100,150,200,250,300]
}

# 网格搜索
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=3,
                           verbose=1)

grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 测试集预测
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 分类报告
report = classification_report(y_test, y_pred)

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('roc_fallback.png')
print(f"已保存图像")

# 输出结果
print("最佳参数组合:")
print(grid_search.best_params_)
print("\n评估指标:")
print(f"准确度: {accuracy:.4f}")
print(f"精确度: {precision:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC值: {roc_auc:.4f}")
print("\n分类报告:")
print(report)