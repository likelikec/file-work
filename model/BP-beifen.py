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

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy

pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FEATURE_COLUMNS=['COM_RAT', 'Jm', 'TCOM_RAT', 'Cyclic', 'jf', 'DPT*',
                   'NOC', 'Inner', 'NTP', 'CONS', 'TODO', 'LCOM', 'CLOC',
                   'CBO', 'PDpt', 'E', 'B', 'D', 'MPC',
                   'Dcy*', 'DIT', 'NOIC', 'CSA', 'Level', 'NOOC', 'OCavg',
                   'OPavg', 'OCmax', 'PDcy', 'Dcy', 'Command', 'n']

LABEL_COLUMN = "1适合LLM"



def load_preprocessed_data(train_path, test_path):
    """加载预处理后的训练集和测试集，并清理缺失值"""
    # 读取原始数据
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    # 清理空值（处理空字符串和NaN）
    def clean_nan_and_empty(df):
        # 将空字符串、纯空格等替换为NaN（覆盖所有列）
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        # 删除包含任何NaN的行
        df = df.dropna()
        # 重置索引保证连续性
        return df.reset_index(drop=True)

    # 执行清理操作
    train_df = clean_nan_and_empty(train_df)
    test_df = clean_nan_and_empty(test_df)

    # 验证列名是否存在（清理后二次校验）
    assert all(col in train_df.columns for col in FEATURE_COLUMNS + [LABEL_COLUMN]), "训练集列名不匹配"
    assert all(col in test_df.columns for col in FEATURE_COLUMNS + [LABEL_COLUMN]), "测试集列名不匹配"

    # 分割特征和标签
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    """构建神经网络模型"""
    model = Sequential([
        Dense(256, activation='relu',
              kernel_regularizer=l2(0.001),
              kernel_initializer='he_normal',
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 2
        Dense(64, activation='relu',
              kernel_regularizer=l2(0.005),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.1),

        # Layer 3
        Dense(64, activation='relu',
              kernel_regularizer=l2(0.005),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=BinaryCrossentropy(),
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
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def main():
    # method_project_split()
    try:
        train_path = r"C:\Users\17958\Desktop\sym-project_train.xlsx"
        test_path = r"C:\Users\17958\Desktop\sym-project_test.xlsx"
        X_train, X_test, y_train, y_test = load_preprocessed_data(train_path, test_path)

        # from imblearn.over_sampling import SMOTE
        # print("\n=== 执行SMOTE过采样 ===")
        # print("原始类别分布:", y_train.value_counts())

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print("过采样后分布:", pd.Series(y_train_res).value_counts())

        # 修改步骤2：使用过采样后的数据划分验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_res, y_train_res,  # 使用过采样后的数据
            test_size=0.25,
            stratify=y_train_res,  # 分层抽样
            random_state=42
        )


        #构建并训练模型
        model = build_model(X_train.shape[1])
        print("\n模型结构摘要：")
        model.summary()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=128,
            callbacks=[early_stop],
            verbose=1
        )

        # 寻找最佳分类阈值
        optimal_threshold = find_optimal_threshold(model, X_val, y_val)
        print(f"\n最佳分类阈值: {optimal_threshold:.4f}")


        y_proba = model.predict(X_test)
        y_pred = (y_proba > optimal_threshold).astype(int)

        # 计算评估指标
        metrics = {
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred),
            '召回率': recall_score(y_test, y_pred),
            'F1分数': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba)
        }

        # 可视化结果
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('训练轨迹')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title('混淆矩阵')

        plt.tight_layout()
        plt.savefig('performance.png')
        plt.close()

        # 打印评估结果
        print("\n=== 评估结果 ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 保存模型
        model.save('trained_model.h5')
        print("模型已保存为 'trained_model.h5'")

    except Exception as e:
        print(f"运行异常: {str(e)}")

if __name__ == "__main__":
    main()
