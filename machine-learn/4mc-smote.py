import pandas as pd
import numpy as np
import os
import joblib  # 用于模型保存与加载
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  # 导入 SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端以保存图片
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义特征列名
FEATURES = ['B', 'COM_RAT', 'Cyclic', 'D',
            'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
            'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT',
            'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
            'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n',
            'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
            'Query', 'RFC', 'TODO', "String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]


def clean_data(df):
    """数据清洗函数"""
    df_clean = df.copy()

    # 过滤目标列，仅保留值为 0 或 1 的行
    df_clean = df_clean[df_clean["1适合LLM"].isin([0, 1])]

    # 将特征列转换为数值类型，转换失败的值置为 NaN
    for col in FEATURES:
        df_clean.loc[:, col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 删除特征列中缺失值超过 2 个的行
    df_clean = df_clean.dropna(
        subset=FEATURES,
        thresh=len(FEATURES) - 2
    )

    # 使用中位数填充剩余的缺失值
    imputer = SimpleImputer(strategy='median')
    df_clean[FEATURES] = imputer.fit_transform(df_clean[FEATURES])

    return df_clean


def save_trained_model(model, model_name, output_dir="trained_models"):
    """保存训练好的模型到文件"""
    # 创建保存目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建模型保存路径
    model_path = os.path.join(output_dir, f"{model_name}_smote_model.pkl")

    # 保存模型
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")


def main():
    try:
        # 数据加载与清洗
        df1 = pd.read_excel(r"C:\Users\17958\Desktop\sym-train01.xlsx")
        df2 = pd.read_excel(r"C:\Users\17958\Desktop\sym-test01.xlsx")
        df_clean1 = clean_data(df1)
        df_clean2 = clean_data(df2)

        print("目标列分布:\n", df_clean1["1适合LLM"].value_counts())

        # 准备数据
        X_train = df_clean1[FEATURES]
        y_train = df_clean1["1适合LLM"].astype(int)

        X_test = df_clean2[FEATURES]
        y_test = df_clean2["1适合LLM"].astype(int)

        # 应用 SMOTE 过采样
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 输出过采样后训练集的样本数量和目标列分布
        print(f"过采样后训练集样本数量: {len(X_train_resampled)}")
        print("过采样后训练集目标列分布:\n", pd.Series(y_train_resampled).value_counts())

        # 将过采样后的训练集保存到 Excel 文件
        train_resampled = pd.DataFrame(X_train_resampled, columns=FEATURES)
        train_resampled["1适合LLM"] = y_train_resampled
        train_resampled.to_excel("train_resampled.xlsx", index=False)

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)

        # 保存标准化器（后续预测时需要使用相同的标准化器）
        save_trained_model(scaler, "standard_scaler")

        # 定义模型及其参数网格
        models = [
            ("SVM", GridSearchCV(
                SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
                param_grid={
                    'C': [16, 17, 19, 18, 20, 21, 30, 35, 40],
                    'gamma': [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 'scale']
                },
                cv=3,
                scoring='f1'
            )),
            ("Decision Tree", GridSearchCV(
                DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                param_grid={
                    'max_depth': [5, 6, 7, 8],
                    'min_samples_split': [6, 7, 8, 9, 10]
                },
                cv=3,
                scoring='f1'
            )),
            ("Random Forest", GridSearchCV(
                RandomForestClassifier(class_weight='balanced', random_state=42),
                param_grid={
                    'n_estimators': [80, 85, 90, 95, 100],
                    'max_depth': [4, 5, 6, 7, 8],
                    'min_samples_split': [3, 4, 5]
                },
                cv=3,
                scoring='f1'
            )),
            ("XGBoost", GridSearchCV(
                XGBClassifier(eval_metric='logloss',
                              scale_pos_weight=np.sum(y_test == 0) / np.sum(y_test == 1),
                              random_state=42),
                param_grid={
                    'learning_rate': [0.22, 0.23, 0.24, 0.25, 0.26],
                    'max_depth': [3, 4, 2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                cv=3,
                scoring='f1'
            ))
        ]

        # 训练和评估模型
        results = []
        for name, model in models:
            try:
                # SVM 使用标准化后的数据，其他模型使用原始数据
                train_data = X_train_scaled if name == "SVM" else X_train_resampled.values
                test_data = X_test_scaled if name == "SVM" else X_test.values

                # 网格搜索训练模型
                model.fit(train_data, y_train_resampled)

                # 输出最佳参数
                print(f"\n=== {name} 最佳参数 ===")
                print(model.best_params_)

                # 预测和评估
                y_pred = model.predict(test_data)
                y_proba = model.predict_proba(test_data)[:, 1]

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                # 对于随机森林，绘制特征重要性图
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
                    plt.close()

                # 保存评估结果
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

                # 保存最佳模型（网格搜索后的最佳估计器）
                save_trained_model(model.best_estimator_, name)

            except Exception as e:
                print(f"{name} 训练失败: {str(e)}")

        # 输出评估结果
        print("\n" + "=" * 50)
        for res in results:
            print(f"\n=== {res['Model']} ===")
            print(f"准确率: {res['Accuracy']:.4f} | 精确率: {res['Precision']:.4f}")
            print(f"召回率: {res['Recall']:.4f} | F1分数: {res['F1']:.4f}")
            print(f"AUC值: {res['AUC']:.4f}")
            print("分类报告:\n", res['Report'])

        # 可视化模型性能
        if results:
            def plot_detailed_metrics(results):
                plt.figure(figsize=(16, 6))

                # 绘制柱状图对比性能指标
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

                # 绘制 ROC 曲线
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
                plt.savefig('detailed_metrics-SMOTE.png', dpi=300)
                plt.close()

            plot_detailed_metrics(results)

    except Exception as e:
        print(f"程序异常: {str(e)}")
        if 'df_clean1' in locals():
            print("清洗后训练集数据样例:\n", df_clean1[FEATURES].head())
        if 'df_clean2' in locals():
            print("清洗后测试集数据样例:\n", df_clean2[FEATURES].head())


if __name__ == "__main__":
    main()