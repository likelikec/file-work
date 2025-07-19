# 导入所有必要库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import rmshap

# ---------------------- 数据读取与预处理 ----------------------
# 读取数据（请确保文件路径正确）
df = pd.read_excel(r"C:\Users\17958\Desktop\symtrain-比例.xlsx", engine='openpyxl')

# 预处理：删除因变量为2的行（假设'1适合LLM'是二分类，值为0/1，删除异常值2）
df = df[df['1适合LLM'] != 2].copy()  # 使用copy避免SettingWithCopyWarning
df['1适合LLM'] = df['1适合LLM'].astype(int)  # 确保标签为整数

# 定义特征列表（请核对特征名是否与数据完全一致，注意可能的拼写错误如"Leve+l"）
features = [
    'B', 'COM_RAT', 'Cyclic', 'D', 'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM',
    'Level', 'LOC', 'N', 'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt',
    'STAT', 'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
    'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n', 'NAAC',
    'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax', 'Query', 'RFC', 'TODO',
    "String processing", "File operations", "Network communication", "Database operations",
    "Mathematical calculation", "User Interface", "Business Logic", "Data Structures and Algorithms",
    "Systems and Tools", "Concurrency and Multithreading", "Exception handling"
]

# 提取特征和标签
X = df[features]
y = df['1适合LLM']

# 处理缺失值（用0填充）
X = X.fillna(0)
y = y.fillna(0)  # 理论上标签不应有缺失，此处为防御性代码

# 划分训练集和测试集（分层抽样保持类别分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42  # 固定随机种子保证可复现
)

# ---------------------- 随机森林网格搜索训练 ----------------------
# 定义网格搜索参数（根据计算资源调整范围）
param_grid = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 分裂所需最小样本数
    'min_samples_leaf': [1, 2, 4],    # 叶节点最小样本数
    'class_weight': [None, 'balanced']  # 处理类别不平衡（若存在）
}

# 初始化随机森林分类器
rf = RandomForestClassifier(
    random_state=42,  # 固定随机种子
    n_jobs=-1         # 使用所有CPU核心
)

# 网格搜索（5折交叉验证）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,             # 5折交叉验证
    n_jobs=-1,        # 并行计算
    scoring='f1',     # 用F1分数作为优化指标（适合不平衡数据）
    verbose=2         # 输出详细日志
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳结果
print("="*50)
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
print("="*50)

# 获取最佳模型
best_rf = grid_search.best_estimator_

# 测试集评估
y_pred = best_rf.predict(X_test)
print("\n测试集评估结果：")
print(classification_report(y_test, y_pred))
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)

# ---------------------- SHAP值计算与可视化 ----------------------
# 初始化SHAP解释器（TreeExplainer对树模型优化）
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)  # 计算SHAP值（二分类返回两个数组）

# ---------------------- 可视化1：全局特征重要性（条形图） ----------------------
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values[1],  # 正类（标签1）的SHAP值
    X_test,
    feature_names=X.columns,
    plot_type="bar",  # 条形图显示特征重要性
    title="全局特征重要性（正类）",
    show=False
)
plt.tight_layout()
plt.show()

# ---------------------- 可视化2：特征影响分布（蜂群图） ----------------------
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values[1],
    X_test,
    feature_names=X.columns,
    plot_type="dot",  # 蜂群图显示特征影响方向和大小
    title="特征对正类预测的影响分布",
    show=False
)
plt.tight_layout()
plt.show()

# ---------------------- 可视化3：单个样本决策路径（依赖图） ----------------------
# 选择第一个测试样本（可替换为任意索引）
sample_idx = 0
sample = X_test.iloc[[sample_idx]]  # 保持二维格式

plt.figure(figsize=(14, 8))
shap.decision_plot(
    explainer.expected_value[1],  # 正类的基准概率
    shap_values[1][sample_idx],   # 该样本的SHAP值
    sample,                       # 样本特征值
    feature_names=X.columns,      # 特征名称
    feature_order='hclust',       # 按特征相关性排序
    title=f"样本 {sample_idx} 的预测决策路径（正类概率）",
    show=False
)
plt.tight_layout()
plt.show()

# ---------------------- 可视化4：特征交互依赖（散点图） ----------------------
# 选择与目标变量相关性最高的特征（示例选前2个）
top_features = shap.utils.top_features(X_test, shap_values[1], n=2)
feature1, feature2 = top_features[0], top_features[1]

plt.figure(figsize=(12, 8))
shap.dependence_plot(
    (feature1, feature2),  # 特征对
    shap_values[1],        # SHAP值
    X_test,                # 原始特征值
    feature_names=X.columns,
    interaction_index='auto',  # 自动检测交互特征
    title=f"{feature1} 与 {feature2} 的SHAP依赖关系",
    show=False
)
plt.tight_layout()
plt.show()