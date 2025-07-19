import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 读取数据
file_path = r"C:\Users\17958\Desktop\train2.0_processed.xlsx"
data = pd.read_excel(file_path)

# 准备特征和目标变量
X = data[[
    "WMC", "DIT", "NOC",  "B", "D", "N",
    "n", "V",  "NCLOC", "LOC", "Cyclic",  "Dcy*",
    "DPT*", "PDcy", "PDpt","NOIC", "Level", "INNER", "OCmax",
     "OSmax",  "STAT", "SUB", "TCOM_RAT"
]]
y = data['1适合LLM']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 定义参数网格
param_grid = {
    'max_depth': [ 2,3,5,10,12,13, None],
    'min_samples_split': [2,3,4,5, 6, 10],
    'min_samples_leaf': [1, 2,3, 4],
    'criterion': ['gini', 'entropy']
}

# 创建决策树分类器并进行网格搜索
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算总体准确度
accuracy = accuracy_score(y_test, y_pred)
print("\n总体准确度 (Accuracy): {:.4f}".format(accuracy))

# 获取详细的分类报告（包含每个类别的指标）
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['负样本 (0)', '正样本 (1)']))

# 计算AUC和ROC曲线
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC值: {:.4f}".format(auc))

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC曲线 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend()
plt.grid(True)
plt.show()

# 输出正负样本的分布
print("\n测试集样本分布:")
print("负样本 (0) 数量:", sum(y_test == 0))
print("正样本 (1) 数量:", sum(y_test == 1))