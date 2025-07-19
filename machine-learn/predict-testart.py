import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import joblib
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型保存目录
MODEL_DIR = 'best-model'
# 预测结果保存目录
PREDICTION_DIR = 'predictions-testart'

# 直接定义特征列表，无需从文件加载
FEATURES = ["COM_RAT", "Cyclic", "Dc+y", "DP+T", "LCOM", "Level", "INNER", "jf",
            "Leve+l", "String processing", "PDpt", "CLOC", "JLOC", "Jm", "Business Logic"]


def clean_data(df, features):
    """安全的数据清洗方法"""
    df_clean = df.copy()

    # 过滤目标列
    df_clean = df_clean[df_clean["1适合LLM"].isin([0, 1])]

    # 类型转换
    for col in features:
        df_clean.loc[:, col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 缺失值处理
    df_clean = df_clean.dropna(
        subset=features,
        thresh=len(features) - 2
    )

    return df_clean


def load_model(model_path):
    """从文件加载模型"""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"模型已从 {model_path} 加载")
        return model
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")


def save_predictions(df_original, y_true, y_pred, model_name, dataset_type, prediction_dir=PREDICTION_DIR):
    """保存预测结果到文件"""
    # 创建保存目录（如果不存在）
    os.makedirs(prediction_dir, exist_ok=True)

    # 创建预测结果DataFrame
    result_df = pd.DataFrame({
        '真实标签': y_true,
        '预测标签': y_pred
    })

    # 合并原始数据（保留原始索引）
    result_with_original = pd.concat([df_original.reset_index(drop=True), result_df], axis=1)

    # 保存到CSV文件
    file_path = os.path.join(prediction_dir, f'{model_name}_{dataset_type}_predictions.csv')
    result_with_original.to_csv(file_path, index=False, encoding='utf-8-sig')

    print(f"预测结果已保存到 {file_path}")
    return file_path


def main():
    try:
        # 定义模型和数据集类型
        model_types = ["Random Forest", "XGBoost"]
        dataset_types = ["sym", "testart"]
        models = {}

        # 加载模型
        for model_type in model_types:
            for dataset_type in dataset_types:
                model_name = f"{model_type}_{dataset_type}"
                model_path = os.path.join(MODEL_DIR, f'{model_type}-{dataset_type}.pkl')
                models[model_name] = load_model(model_path)

        # 加载测试数据 - 请修改为实际测试集路径
        test_data_path = r"C:\Users\17958\Desktop\testart-test01.xlsx"
        df_test = pd.read_excel(test_data_path)

        original_test_size = len(df_test)
        df_test_clean = clean_data(df_test, FEATURES)

        print(f"测试集清洗完成，原始数据量: {original_test_size}，处理后数据量: {len(df_test_clean)}")
        print("测试集目标列分布:\n", df_test_clean["1适合LLM"].value_counts())

        # 准备数据
        X_test = df_test_clean[FEATURES].values
        y_test = df_test_clean["1适合LLM"].astype(int)

        # 评估模型
        results = []

        for model_name, model in models.items():
            try:
                model_type, dataset_type = model_name.split('_')
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # 保存预测结果
                save_predictions(df_test_clean, y_test, y_pred, model_type, dataset_type)

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "AUC": roc_auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "Report": classification_report(y_test, y_pred)
                })

                print(f"\n=== {model_name} 模型评估 ===")
                print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
                print(f"精确率: {precision:.4f} | 召回率: {recall:.4f} | F1分数: {f1:.4f}")
                print(f"AUC: {roc_auc:.4f}")
                print("分类报告:")
                print(classification_report(y_test, y_pred))

            except Exception as e:
                print(f"评估 {model_name} 模型时出错: {str(e)}")

        # 可视化评估结果
        if results:
            plt.figure(figsize=(12, 6))

            # 绘制ROC曲线
            plt.subplot(1, 2, 1)
            for res in results:
                plt.plot(res['fpr'], res['tpr'],
                         label=f"{res['Model']} (AUC = {res['AUC']:.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('模型ROC曲线对比')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            # 绘制性能指标对比图
            plt.subplot(1, 2, 2)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            x = np.arange(len(results))
            width = 0.15

            for i, metric in enumerate(metrics):
                plt.bar(x + i * width, [res[metric] for res in results], width, label=metric)

            plt.xlabel('模型')
            plt.ylabel('分数')
            plt.title('模型性能对比')
            plt.xticks(x + width * (len(metrics) - 1) / 2, [res['Model'] for res in results], rotation=45)
            plt.ylim(0, 1.05)
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig('model_evaluation.png', dpi=300)
            plt.close()

            print("\n模型评估可视化结果已保存为 'model_evaluation.png'")

    except Exception as e:
        print(f"程序异常: {str(e)}")
        if 'df_test_clean' in locals():
            print("测试集清洗后数据样例:\n", df_test_clean[FEATURES].head())


if __name__ == "__main__":
    main()