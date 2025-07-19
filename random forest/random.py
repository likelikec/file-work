import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# 设置文件路径
train_file = r'C:\Users\17958\Desktop\比例_train.xlsx'
test_file = r'C:\Users\17958\Desktop\比例_test.xlsx'
output_feature_importance = r'C:\Users\17958\Desktop\特征重要性.xlsx'

# 定义特征列和目标列
feature_columns = ['B', 'COM_RAT', 'Cyclic', 'D', 'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
                   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'TCOM_RAT', 'V', 'WMC', 'CBO', 
                   'CLOC', 'Command', 'CONS', 'CSA', 'CSO', 'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 
                   'MPC', 'n', 'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax', 'Query', 'RFC', 'TODO', 
                   "String processing", "File operations", "Network communication", "Database operations", "Mathematical calculation", 
                   "User Interface", "Business Logic", "Data Structures and Algorithms", "Systems and Tools", 
                   "Concurrency and Multithreading", "Exception handling"]
target_column = '1适合LLM'

# 读取数据
try:
    train_data = pd.read_excel(train_file)
    test_data = pd.read_excel(test_file)
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
except Exception as e:
    print(f"读取数据出错: {e}")
    exit()

# 准备特征和目标变量
X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]

# 创建并训练随机森林模型
model = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=200,
    random_state=42
)

# 训练模型
try:
    model.fit(X_train, y_train)
    print("模型训练完成")
except Exception as e:
    print(f"模型训练出错: {e}")
    exit()

# 评估模型
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# 获取特征重要性
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
})

# 按重要性排序
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 保存特征重要性到Excel
try:
    feature_importance.to_excel(output_feature_importance, index=False)
    print(f"特征重要性已保存到 {output_feature_importance}")
except Exception as e:
    print(f"保存特征重要性出错: {e}")

# 保存模型
import joblib
model_path = r'C:\Users\17958\Desktop\random_forest_model.pkl'
try:
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")
except Exception as e:
    print(f"保存模型出错: {e}")    