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


# #SHAP值
# FEATURE_COLUMNS=["Level*", "OCavg", "TCOM_RAT", "MPC", "DPT", "D", "LOC", "Dcy", "OSavg", "OSmax","DPT*","CONS","Query"
#                  ,"Level","STAT","NAAC","Cyclic"]



# #XGBoost信息增益
# FEATURE_COLUMNS=["Level*", "LOC", "NTP", "CONS", "OCavg", "NAIC", "INNER", "STAT",
#                  "TODO", "N", "D", "DPT", "WMC", "Level", "NOOC", "TCOM_RAT", "CSOA", "CSA", "NOAC", "SUB"]


# # 聚类
# FEATURE_COLUMNS=['Cyclic', 'TCOM_RAT', 'DPT',
#                   'CLOC', 'PDpt', 'NOC',
#                  'Inner', 'LCOM',
#                  'Dcy*', 'DIT', 'E', 'B',
#                   'Level', 'OCavg',
#                  ]

# #
# FEATURE_COLUMNS=['COM_RAT', 'Jm', 'TCOM_RAT', 'Cyclic', 'jf', 'DPT*',
#                    'NOC', 'Inner', 'NTP', 'CONS', 'TODO', 'LCOM', 'CLOC',
#                    'CBO', 'PDpt', 'E', 'B', 'D', 'MPC',
#                    'Dcy*', 'DIT', 'NOIC', 'CSA', 'Level', 'NOOC', 'OCavg',
#                    'OPavg', 'OCmax', 'PDcy', 'Dcy', 'Command', 'n']
#全集
FEATURE_COLUMNS = ['B', 'COM_RAT', 'Cyclic', 'D',
   'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
   'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
   'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
   'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
   'Query', 'RFC', 'TODO',"String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]


# cv系数
# FEATURE_COLUMNS = [
#        'Cyclic', 'D',
#    'Dcy*',  'DPT*', 'LCOM',
#       'NOIC',
#    'TCOM_RAT',    'CSA',
#     'Dcy', 'DPT',   'MPC',
#       'OPavg',
#
# ]
LABEL_COLUMN = "1适合LLM"

# feature_columns =  [
#     "metrics","class",'B', 'COM_RAT', 'Cyclic', 'D',
#    'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
#    'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
#    'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
#    'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
#    'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
#    'Query', 'RFC', 'TODO'
# ]
# label_column = "1适合LLM"
# file_path = r"C:\Users\17958\Desktop\train_4.0.xlsx"

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
        # 第一隐藏层（对应layer1）
        Dense(128, activation='relu', kernel_regularizer=l2(0.02), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),

        # 第二隐藏层（对应layer2，移除残差连接）
        Dense(32, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.1),

        # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        # BatchNormalization(),
        # Dropout(0.2),

        # 第三隐藏层（对应layer3）
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        #
        # # 第四隐藏层（对应layer4，移除残差连接）
        Dense(64, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),

        # 输出层
        Dense(1, activation='sigmoid')  # 自动处理logits转换
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
# def method_project_split():
#     # 读取并预处理数据
#     df = pd.read_excel(file_path)
#     # 获取所有项目名称
#     projects = df['metrics'].unique()
#     print(f"发现 {len(projects)} 个项目: {projects}")
#
#     # 初始化存储容器
#     all_train = pd.DataFrame()
#     all_test = pd.DataFrame()
#
#     # 遍历每个项目进行划分
#     for project in projects:
#         # 提取当前项目数据
#         project_df = df[df['metrics'] == project]
#
#         # 分层划分8-2
#         X = project_df[feature_columns]
#         y = project_df[label_column]
#
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y,
#             test_size=0.2,
#             stratify=y,
#             random_state=42
#         )
#
#         # 标准化处理（仅在特征列上执行）
#         scaler = StandardScaler()
#         # 提取需要标准化的特征列
#         train_features = X_train[FEATURE_COLUMNS]
#         test_features = X_test[FEATURE_COLUMNS]
#
#         # 训练集拟合并转换，测试集仅转换
#         X_train[FEATURE_COLUMNS] = scaler.fit_transform(train_features)
#         X_test[FEATURE_COLUMNS] = scaler.transform(test_features)
#
#         # 合并当前项目的划分结果
#         train_df = pd.concat([X_train, y_train], axis=1)
#         test_df = pd.concat([X_test, y_test], axis=1)
#
#         all_train = pd.concat([all_train, train_df])
#         all_test = pd.concat([all_test, test_df])
#
#         # 验证数据完整性
#     print(f"\n总训练集样本数: {len(all_train)} | 总测试集样本数: {len(all_test)}")
#     print(f"原始数据总样本数: {len(df)} | 合并后总数: {len(all_train) + len(all_test)}")
#
#     # 保存结果（标准化后的数据）
#     all_train.to_excel(r"C:\Users\17958\Desktop\project_train.xlsx", index=False)
#     all_test.to_excel(r"C:\Users\17958\Desktop\project_test.xlsx", index=False)

def main():
    # method_project_split()
    try:
        train_path = r"C:\Users\17958\Desktop\sym-project_train.xlsx"
        test_path = r"C:\Users\17958\Desktop\sym-project_test.xlsx"
        X_train, X_test, y_train, y_test = load_preprocessed_data(train_path, test_path)

        # from imblearn.over_sampling import SMOTE
        # print("\n=== 执行SMOTE过采样 ===")
        # print("原始类别分布:", y_train.value_counts())
        #
        # smote = SMOTE(random_state=42)
        # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        # print("过采样后分布:", pd.Series(y_train_res).value_counts())

        # 修改步骤2：使用过采样后的数据划分验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,  # 使用过采样后的数据
            test_size=0.25,
            stratify=y_train,  # 分层抽样
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
