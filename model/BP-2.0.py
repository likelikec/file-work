import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import BinaryCrossentropy
import random
from collections import defaultdict
import json
import math

pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
FEATURE_COLUMNS = ['COM_RAT', 'Jm', 'TCOM_RAT', 'Cyclic', 'jf', 'DPT*',
                   'NOC', 'Inner', 'NTP', 'CONS', 'TODO', 'LCOM', 'CLOC',
                   'CBO', 'PDpt', 'E', 'B', 'D', 'MPC',
                   'Dcy*', 'DIT', 'NOIC', 'CSA', 'Level', 'NOOC', 'OCavg',
                   'OPavg', 'OCmax', 'PDcy', 'Dcy', 'Command', 'n'] # 保持原有特征列
LABEL_COLUMN = "1适合LLM"

def find_optimal_threshold(model, X_val, y_val):
    """寻找最佳分类阈值"""
    y_proba = model.predict(X_val, verbose=0)
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-7)
    return thresholds[np.argmax(f1_scores)]
def load_preprocessed_data(train_path, test_path):
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
    # 保持原有数据加载逻辑


def log_uniform(minval, maxval):
    """对数均匀分布采样"""
    return lambda: math.exp(random.uniform(math.log(minval), math.log(maxval)))


class AdvancedHyperparamGenerator:
    """高级超参数生成器"""

    def __init__(self):
        self.param_space = {
            'n_layers': lambda: random.randint(3, 6),  # 3-6个隐藏层
            'units': lambda: int(random.choice([
                64, 128, 256, 512,
                random.randint(50, 100),
                random.randint(100, 200)
            ])),
            'dropout': lambda: round(random.uniform(0.1, 0.7), 1),
            'l2_reg': log_uniform(1e-5, 1e-2),
            'activation': lambda: random.choice(['relu', 'elu', 'selu', 'tanh']),
            'batch_size': lambda: random.choice([32, 64, 128, 256]),
            # 'optimizer': lambda: random.choice([
            #     ('adam', {'learning_rate': log_uniform(1e-4, 1e-2)()}),
            #     ('rmsprop', {'learning_rate': log_uniform(1e-4, 1e-2)(), 'rho': 0.9}),
            #     ('nadam', {'learning_rate': log_uniform(1e-4, 1e-2)()})
            # ]),
            'use_batchnorm': lambda: random.choice([True, False]),
            'lr_schedule': lambda: random.choice([True, False])
        }

    def generate_config(self):
        """生成完整超参数配置"""
        config = {
            'layers': [],
            'global': {}
        }

        # 生成网络结构配置
        n_layers = self.param_space['n_layers']()
        for _ in range(n_layers):
            layer_config = {
                'units': self.param_space['units'](),
                'dropout': self.param_space['dropout'](),
                'l2_reg': self.param_space['l2_reg'](),
                'activation': self.param_space['activation'](),
                'batchnorm': self.param_space['use_batchnorm']()
            }
            config['layers'].append(layer_config)

        # 生成全局配置
        config['global']['batch_size'] = self.param_space['batch_size']()
        config['global']['optimizer'] = self.param_space['optimizer']()
        config['global']['lr_schedule'] = self.param_space['lr_schedule']()

        return config


def build_advanced_model(input_dim, config):
    """根据配置构建高级模型"""
    model = Sequential()

    # 输入层
    model.add(Dense(
        config['layers'][0]['units'],
        activation=config['layers'][0]['activation'],
        kernel_regularizer=l1_l2(l2=config['layers'][0]['l2_reg']),
        input_shape=(input_dim,)
    ))
    if config['layers'][0]['batchnorm']:
        model.add(BatchNormalization())

    # 添加隐藏层
    for layer_config in config['layers'][1:]:
        model.add(Dense(
            layer_config['units'],
            activation=layer_config['activation'],
            kernel_regularizer=l1_l2(l2=layer_config['l2_reg'])
        ))
        if layer_config['batchnorm']:
            model.add(BatchNormalization())
        model.add(Dropout(layer_config['dropout']))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 配置优化器
    opt_name, opt_params = config['global']['optimizer']
    optimizer = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'nadam': Nadam
    }[opt_name](**opt_params)

    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def create_callbacks(config):
    """创建动态回调函数"""
    callbacks = [EarlyStopping(
        monitor='val_auc',
        patience=20,
        mode='max',
        restore_best_weights=True
    )]

    if config['global']['lr_schedule']:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ))
    return callbacks


def main():
    try:
        # 数据加载和预处理
        train_path = r"C:\Users\17958\Desktop\project_train.xlsx"
        test_path = r"C:\Users\17958\Desktop\project_test.xlsx"
        X_train, X_test, y_train, y_test = load_preprocessed_data(train_path, test_path)

        # 特征工程
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # 数据分割
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # 超参数搜索配置
        hyper_generator = AdvancedHyperparamGenerator()
        best_score = -np.inf
        best_model = None
        best_config = None
        search_iterations = 50
        history = []

        for i in range(search_iterations):
            print(f"\n=== 迭代 {i + 1}/{search_iterations} ===")

            # 生成新配置
            config = hyper_generator.generate_config()

            try:
                # 构建和训练模型
                model = build_advanced_model(X_train_final.shape[1], config)
                callbacks = create_callbacks(config)

                history = model.fit(
                    X_train_final, y_train_final,
                    validation_data=(X_val, y_val),
                    epochs=300,
                    batch_size=config['global']['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )

                # 评估模型
                y_proba = model.predict(X_val, verbose=0)
                score = roc_auc_score(y_val, y_proba)

                # 保存最佳模型
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_config = config
                    print(f"新最佳 AUC: {score:.4f}")
                    print(f"配置摘要：")
                    print(f"优化器: {config['global']['optimizer'][0]}")
                    print(f"学习率: {config['global']['optimizer'][1]['learning_rate']:.5f}")
                    print(f"批大小: {config['global']['batch_size']}")
                    print(f"层数: {len(config['layers'])}")

            except Exception as e:
                print(f"跳过无效配置: {str(e)}")
                continue

        # 最终评估
        print("\n=== 最优配置评估 ===")
        y_proba = best_model.predict(X_test, verbose=0)
        optimal_threshold = find_optimal_threshold(best_model, X_val, y_val)
        y_pred = (y_proba > optimal_threshold).astype(int)

        # 保存结果
        best_model.save("optimized_model.h5")
        with open("best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)

        # 可视化报告
        plt.figure(figsize=(18, 6))

        # 训练过程可视化
        plt.subplot(1, 3, 1)
        plt.plot(history.history['auc'], label='训练AUC')
        plt.plot(history.history['val_auc'], label='验证AUC')
        plt.title('训练过程')
        plt.legend()

        # 特征重要性（基于权重）
        plt.subplot(1, 3, 2)
        weights = best_model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        sns.barplot(x=importance, y=FEATURE_COLUMNS)
        plt.title('特征重要性')

        # 混淆矩阵
        plt.subplot(1, 3, 3)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title('测试集混淆矩阵')

        plt.tight_layout()
        plt.savefig('advanced_performance.png')

        # 打印评估指标
        print("\n=== 最终测试结果 ===")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"运行错误: {str(e)}")


if __name__ == "__main__":
    main()