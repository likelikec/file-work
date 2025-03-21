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
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
# 配置显示选项
pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data():
    """数据加载与预处理"""
    try:
        # 加载数据
        df = pd.read_excel(r"C:\Users\17958\Desktop\train.xlsx")

        # 定义特征和目标列
        features = ['Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*',
                    'PDcy', 'PDpt', 'OCavg', 'OCmax',
                    'WMC', 'CLOC', 'JLOC', 'LOC']
        target = '1适合LLM'

        # 数据清洗
        df = df[features + [target]].dropna(subset=[target])
        df[target] = df[target].astype(int)

        # 处理缺失值
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])

        return df[features], df[target]

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def build_model(input_dim):
    """修正后的模型构建函数"""
    model = Sequential([
        Dense(128, activation='relu',
              kernel_regularizer=l2(0.001),
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu',
              kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def find_optimal_threshold(model, X_val, y_val):
    y_proba = model.predict(X_val)
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    # 寻找最佳F1阈值
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def main():
    try:
        # 数据加载与预处理
        df = pd.read_excel(r"C:\Users\17958\Desktop\train.xlsx")
        features = ['Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*', 'PDcy', 'PDpt',
                    'OCavg', 'OCmax', 'WMC', 'CLOC', 'JLOC', 'LOC']
        target = '1适合LLM'

        # 数据清洗
        df = df[features + [target]].dropna(subset=[target])
        df[target] = df[target].astype(int)

        # 处理缺失值
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(df[features])
        y = df[target]

        # SMOTE过采样
        sm = SMOTE(sampling_strategy=0.8, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 修复2：正确调用模型构建
        model = build_model(X_train.shape[1])
        print("\n模型结构摘要：")
        model.summary()

        # 训练配置
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=200,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )

        # 阈值优化
        y_proba = model.predict(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        # 评估
        y_pred = (y_proba > optimal_threshold).astype(int)

        # 计算指标
        metrics = {
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred),
            '召回率': recall_score(y_test, y_pred),
            'F1分数': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba)
        }

        # 可视化
        plt.figure(figsize=(15, 5))

        # 训练曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型训练轨迹')
        plt.legend()

        # 混淆矩阵
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')

        plt.tight_layout()
        plt.savefig('result.png', dpi=300)
        plt.close()

        # 输出结果
        print("\n=== 最终评估结果 ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        print("结果已保存到 results.png")

    except Exception as e:
        print(f"程序异常: {str(e)}")


if __name__ == "__main__":
    main()