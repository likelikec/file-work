import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

try:
    # 读取数据
    file_path = r"C:\Users\17958\Desktop\train_4.0.xlsx"
    data = pd.read_excel(file_path)
    print("数据读取成功，样本数:", len(data))

    # 准备特征和目标变量
    X = data[['B', 'COM_RAT', 'Cyclic', 'D',
   'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
   'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
   'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
   'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
   'Query', 'RFC', 'TODO']]
    y = data['1适合LLM']
    # 使用0填充缺失值
    X = X.fillna(0)
    y = y.fillna(0)
    # 检查数据是否有缺失值
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        raise ValueError("数据中存在缺失值，请先处理！")

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用SMOTE处理不平衡数据
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("SMOTE处理后训练集样本数:", len(X_train_balanced))

    # 定义XGBoost参数网格
    param_grid = {
        'max_depth': [1,2,3, 5, 7,None],
        'learning_rate': [0.1,0.15,0.2,0.21,0.22,0.23,0.24,0.25, 0.3],
        'n_estimators': [100, 150,200,220,240, 300],
        'min_child_weight': [1, 2,3, 5],
        'subsample': [0.8, 0.9,0.92,0.94,0.95,0.96,0.97,0.98,1.0,1.5]
    }

    # 创建XGBoost分类器（移除use_label_encoder）
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # 进行网格搜索
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)

    # 输出最佳参数
    print("最佳参数:", grid_search.best_params_)

    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 计算总体准确度
    accuracy = accuracy_score(y_test, y_pred)
    print("\n总体准确度 (Accuracy): {:.4f}".format(accuracy))

    # 获取详细的分类报告
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

except Exception as e:
    print("发生错误:", str(e))