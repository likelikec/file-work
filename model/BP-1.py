import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
import random
from collections import defaultdict
import json

pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
FEATURE_COLUMNS =['B', 'COM_RAT', 'Cyclic', 'D',
   'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
   'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
   'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
   'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
   'Query', 'RFC', 'TODO']
LABEL_COLUMN = "1适合LLM"


def load_preprocessed_data(train_path, test_path):
    """加载并预处理数据"""
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    def clean_data(df):
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna()
        return df.reset_index(drop=True)

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # 验证数据格式
    assert all(col in train_df.columns for col in FEATURE_COLUMNS + [LABEL_COLUMN]), "训练集列名不匹配"
    assert all(col in test_df.columns for col in FEATURE_COLUMNS + [LABEL_COLUMN]), "测试集列名不匹配"

    return (
        train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS],
        train_df[LABEL_COLUMN], test_df[LABEL_COLUMN]
    )


def build_model(input_dim, layer_configs):
    """动态构建神经网络模型"""
    model = Sequential()

    # 输入层和第一个隐藏层
    first_layer = layer_configs[0]
    model.add(Dense(
        units=first_layer['units'],
        activation='relu',
        kernel_regularizer=l2(first_layer['l2']),
        input_shape=(input_dim,)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(first_layer['dropout']))

    # 后续隐藏层
    for layer in layer_configs[1:]:
        model.add(Dense(
            units=layer['units'],
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(layer['l2'])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(layer['dropout']))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001,clipvalue=1.0),
        loss=BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def generate_random_config(max_layers=5):  # 减少最大层数
    """生成更保守的层配置"""
    num_layers = random.randint(2, max_layers)
    config = []
    for _ in range(num_layers):
        params = {
            'units': random.choice([32, 64, 128,256]),  # 减少神经元数量
            'dropout': round(random.uniform(0.1, 0.5), 1),  # 调整dropout范围
            'l2': random.choice([0.01, 0.02, 0.03])  # 增强正则化
        }
        config.append(params)
    return config


def find_optimal_threshold(model, X_val, y_val):
    """寻找最佳分类阈值"""
    y_proba = model.predict(X_val, verbose=0)
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-7)
    return thresholds[np.argmax(f1_scores)]


def main():
    try:
        # 数据加载
        train_path = r"C:\Users\17958\Desktop\sym-project_train.xlsx"
        test_path = r"C:\Users\17958\Desktop\sym-project_test.xlsx"
        X_train, X_test, y_train, y_test = load_preprocessed_data(train_path, test_path)

        # 数据预处理
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # 处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # 分割验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
        )

        # 超参数搜索配置
        best_metrics = {'f1': 0}
        best_model = None
        best_config = None
        search_iterations = 50
        history_records = []

        # 超参数搜索循环
        for i in range(search_iterations):
            print(f"\n=== 第 {i + 1}/{search_iterations} 次参数搜索 ===")

            # 生成随机配置
            layer_configs = generate_random_config()
            print("当前层配置：")
            for idx, layer in enumerate(layer_configs):
                print(f"  层 {idx + 1}: 单元数={layer['units']}, Dropout={layer['dropout']}, L2={layer['l2']}")

            # 构建并训练模型
            model = build_model(X_train_final.shape[1], layer_configs)
            early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=128,
                callbacks=[early_stop],
                verbose=0
            )

            # 评估当前配置
            optimal_threshold = find_optimal_threshold(model, X_val, y_val)
            y_val_proba = model.predict(X_val, verbose=0)
            y_val_pred = (y_val_proba > optimal_threshold).astype(int)

            current_metrics = {
                'f1': f1_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_proba),
                'recall': recall_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred)
            }
            history_records.append(current_metrics)

            # 更新最佳配置
            if current_metrics['f1'] > best_metrics['f1']:
                best_metrics = current_metrics
                best_model = model
                best_config = layer_configs
                print(f"发现更好配置 ▶ F1: {current_metrics['f1']:.4f} | AUC: {current_metrics['auc']:.4f}")

        # 最终评估和输出
        print("\n=== 最佳配置 ===")
        for idx, layer in enumerate(best_config):
            print(f"  层 {idx + 1}: 单元数={layer['units']}, Dropout={layer['dropout']}, L2={layer['l2']}")

        # 测试集评估
        optimal_threshold = find_optimal_threshold(best_model, X_val, y_val)
        y_proba = best_model.predict(X_test, verbose=0)
        y_pred = (y_proba > optimal_threshold).astype(int)

        # 可视化结果
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot([x['f1'] for x in history_records], label='验证集F1')
        plt.plot([x['auc'] for x in history_records], label='验证集AUC')
        plt.title('超参数搜索过程')
        plt.xlabel('迭代次数')
        plt.ylabel('分数')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title('测试集混淆矩阵')

        plt.tight_layout()
        plt.savefig('performance.png')
        plt.close()

        # 保存结果
        best_model.save('best_model.h5')
        with open('best_config.json', 'w') as f:
            json.dump(best_config, f)

        # 打印评估结果
        print("\n=== 测试集评估结果 ===")
        print(f"最佳阈值: {optimal_threshold:.4f}")
        print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
        print(f"精确率: {precision_score(y_test, y_pred):.4f}")
        print(f"召回率: {recall_score(y_test, y_pred):.4f}")
        print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"运行异常: {str(e)}")


if __name__ == "__main__":
    main()