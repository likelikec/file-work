import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义特征列名
FEATURES = [
    'Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*', 'PDcy', 'PDpt',
    'OCavg', 'OCmax', 'WMC', 'CLOC', 'JLOC', 'LOC', 'JF', 'JM'
]

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
        df = pd.read_excel(r"C:\Users\17958\Desktop\hits类覆盖率+指标.xlsx")
        original_size = len(df)
        df_clean = clean_data(df)

        print(f"数据清洗完成，原始数据量: {original_size}，处理后数据量: {len(df_clean)}")
        print("目标列分布:\n", df_clean["1适合LLM"].value_counts())

        # 准备数据
        X = df_clean[FEATURES]
        y = df_clean["1适合LLM"].astype(int)

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 模型配置
        models = [
            ("SVM", SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')),
            ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight='balanced')),
            ("Random Forest",
             RandomForestClassifier(n_estimators=200,
                                    class_weight='balanced',
                                    min_samples_split=5,
                                    max_depth=5)),
            ("XGBoost", XGBClassifier(eval_metric='logloss', scale_pos_weight=np.sum(y == 0) / np.sum(y == 1)))
        ]

        # 训练评估
        results = []
        for name, model in models:
            try:
                train_data = X_train_scaled if name == "SVM" else X_train.values
                test_data = X_test_scaled if name == "SVM" else X_test.values

                model.fit(train_data, y_train)
                y_pred = model.predict(test_data)
                y_proba = model.predict_proba(test_data)[:, 1]

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

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

        # 结果输出
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
        if 'df_clean' in locals():
            print("清洗后数据样例:\n", df_clean[FEATURES].head())

if __name__ == "__main__":
    main()