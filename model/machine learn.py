import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义特征列名

# FEATURES =['Level*', 'Dcy*', 'Cyclic', 'jf', 'COM_RAT',
#            'DPT*', 'INNER', 'JLOC', 'Level', 'CLOC', 'Jm', 'PDpt', 'LCOM'
#            ,"String processing", "Mathematical calculation", "User Interface",
#             "Business Logic",  "Exception handling"
#            ]

# #testart
# FEATURES=['COM_RAT', 'Cyclic', 'Dc+y', 'DP+T', 'Level', 'INNER', 'jf', 'Leve+l',
#  'String processing', 'LCOM', 'Jm', 'Business Logic', 'PDpt', 'CLOC', 'OCavg']
# #sym
# FEATURES=['Level*', 'Dcy*', 'Cyclic', 'jf', 'COM_RAT', 'DPT*', 'INNER', 'JLOC', 'Level', 'CLOC', 'Jm', 'PDpt', 'LCOM' ,
#           "String processing", "Mathematical calculation", "User Interface", "Business Logic",  "Exception handling"]
# # #交集
# FEATURES=[ 'Dc+y', 'Cyclic', 'jf', 'COM_RAT', 'DP+T', 'INNER',  'Leve+l',
#            'Level' ,'CLOC',  'PDpt', 'Jm','LCOM' ,"String processing", "Business Logic"]
   #交集
FEATURES= ["COM_RAT", "Cyclic", "Dc+y", "DP+T", "LCOM", "Level", "INNER", "jf", "Leve+l", "String processing", "PDpt", "CLOC", "JLOC", "Jm","Business Logic"]

# FEATURES =['B', 'COM_RAT', 'Cyclic', 'D',
#             'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
#             'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT',
#             'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
#             'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n',
#             'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
#             'Query', 'RFC', 'TODO', "String processing", "File operations", "Network communication",
#             "Database operations", "Mathematical calculation", "User Interface",
#             "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
#             "Concurrency and Multithreading", "Exception handling"]


def clean_data(df):
    """安全的数据清洗方法"""
    df_clean = df.copy()

    # 过滤目标列
    df_clean = df_clean[df_clean["1适合LLM"].isin([0, 1])]

    # 类型转换
    for col in FEATURES:
        df_clean.loc[:, col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 缺失值处理
    df_clean = df_clean.dropna(
        subset=FEATURES,
        thresh=len(FEATURES) - 2
    )

    imputer = SimpleImputer(strategy='median')
    df_clean[FEATURES] = imputer.fit_transform(df_clean[FEATURES])

    return df_clean


def main():
    try:
        # 数据加载与清洗
        df_train = pd.read_excel(r"C:\Users\17958\Desktop\sym-train01.xlsx")
        df_test = pd.read_excel(r"C:\Users\17958\Desktop\sym-test01.xlsx")

        original_train_size = len(df_train)
        original_test_size = len(df_test)

        df_train_clean = clean_data(df_train)
        df_test_clean = clean_data(df_test)

        print(f"训练集清洗完成，原始数据量: {original_train_size}，处理后数据量: {len(df_train_clean)}")
        print("训练集目标列分布:\n", df_train_clean["1适合LLM"].value_counts())
        print(f"测试集清洗完成，原始数据量: {original_test_size}，处理后数据量: {len(df_test_clean)}")
        print("测试集目标列分布:\n", df_test_clean["1适合LLM"].value_counts())

        # 准备数据
        X_train = df_train_clean[FEATURES]
        y_train = df_train_clean["1适合LLM"].astype(int)
        X_test = df_test_clean[FEATURES]
        y_test = df_test_clean["1适合LLM"].astype(int)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 模型配置
        models = [
            ("SVM", GridSearchCV(
                SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
                param_grid={
                    'C': [21, 25, 27, 28, 29, 30, 31, 32, 33, 35, 40],

                },
                cv=3,
                scoring='f1'
            )),
            ("Decision Tree", GridSearchCV(
                DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                param_grid={
                    'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split': [4, 5, 6, 7, 8]
                },
                cv=3,
                scoring='f1'
            )),
            ("Random Forest", GridSearchCV(
                RandomForestClassifier(class_weight='balanced', random_state=42),
                param_grid={
                    'n_estimators': [95, 97, 98, 99, 100, 105, 110, 130, 120],
                    'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split': [4, 5, 6, 7, 8, 9]
                },
                cv=3,
                scoring='f1'
            )),
            ("XGBoost", GridSearchCV(
                XGBClassifier(eval_metric='logloss',
                              scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
                              random_state=42),
                param_grid={
                    'learning_rate': [0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
                    'max_depth': [3, 4, 5, 2],
                    'subsample': [0.8, 0.95, 0.9, 1.0]
                },
                cv=3,
                scoring='f1'
            ))
        ]

        # 训练评估
        results = []
        for name, model in models:
            try:
                train_data = X_train_scaled if name == "SVM" else X_train.values
                test_data = X_test_scaled if name == "SVM" else X_test.values

                # 网格搜索训练
                model.fit(train_data, y_train)

                # 输出最佳参数
                print(f"\n=== {name} 最佳参数 ===")
                print(model.best_params_)

                y_pred = model.predict(test_data)
                y_proba = model.predict_proba(test_data)[:, 1]

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                if name == "Random Forest":
                    plt.figure(figsize=(10, 6))
                    importances = model.best_estimator_.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    plt.title("特征重要性 - 随机森林")
                    plt.barh(range(len(indices)), importances[indices], align='center')
                    plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
                    plt.xlabel('相对重要性')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png', dpi=300)

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "AUC": roc_auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "Report": classification_report(y_test, y_pred)
                })

            except Exception as e:
                print(f"{name} 训练失败: {str(e)}")

        print("\n" + "=" * 50)
        for res in results:
            print(f"\n=== {res['Model']} ===")
            print(f"准确率: {res['Accuracy']:.4f} | 精确率: {res['Precision']:.4f}")
            print(f"召回率: {res['Recall']:.4f} | F1分数: {res['F1']:.4f}")
            print(f"AUC值: {res['AUC']:.4f}")
            print("分类报告:\n", res['Report'])

        # 可视化
        if results:
            def plot_detailed_metrics(results):
                plt.figure(figsize=(16, 6))

                plt.subplot(1, 2, 1)
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                bar_width = 0.15
                x = np.arange(len(results))

                for i, metric in enumerate(metrics):
                    values = [res[metric] for res in results]
                    plt.bar(x + i * bar_width, values, bar_width,
                            color=colors[i], label=metric)

                plt.title('模型性能综合对比', fontsize=14)
                plt.xticks(x + bar_width * 2, [res['Model'] for res in results])
                plt.ylabel('Score', fontsize=12)
                plt.ylim(0, 1.05)
                plt.legend(bbox_to_anchor=(1.05, 1))
                plt.grid(axis='y', alpha=0.3)

                plt.subplot(1, 2, 2)
                for res in results:
                    plt.plot(res['fpr'], res['tpr'],
                             label=f"{res['Model']} (AUC={res['AUC']:.2f})")

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC曲线对比', fontsize=14)
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)

                plt.tight_layout()
                plt.savefig('detailed_metrics.png', dpi=300)
                plt.close()

            plot_detailed_metrics(results)

    except Exception as e:
        print(f"程序异常: {str(e)}")
        if 'df_train_clean' in locals():
            print("训练集清洗后数据样例:\n", df_train_clean[FEATURES].head())
        if 'df_test_clean' in locals():
            print("测试集清洗后数据样例:\n", df_test_clean[FEATURES].head())


if __name__ == "__main__":
    main()