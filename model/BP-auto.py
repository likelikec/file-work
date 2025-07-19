import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
# 配置参数
FEATURE_COLUMNS = ['B', 'COM_RAT', 'Cyclic', 'D',
   'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
   'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
   'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
   'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
   'Query', 'RFC', 'TODO']
LABEL_COLUMN = "1适合LLM"

FILE_PATH = r"C:\Users\17958\Desktop\train_4.0.xlsx"

# 超参数搜索配置
MAX_TRIALS = 100  # 最大试验次数
EXECUTIONS_PER_TRIAL = 2  # 每次试验重复次数
MAX_EPOCHS = 400  # 最大训练轮数


def load_preprocessed_data(train_path, test_path):
    """加载预处理后的训练集和测试集"""
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)


    # 清理数据
    def clean_data(df):
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        # 删除包含任何NaN的行
        df = df.dropna()

        return df[FEATURE_COLUMNS + [LABEL_COLUMN]].astype(float)

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # 分割特征和标签
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    return X_train.values, X_test.values, y_train.values, y_test.values


def build_hypermodel(hp):
    """构建可调参的BP神经网络"""
    from tensorflow.keras.metrics import AUC, Precision, Recall  # 新
    model = Sequential()

    # 网络深度配置
    num_layers = hp.Int('num_layers', 2, 5, default=3)

    # 输入层
    model.add(Dense(
        units=hp.Int('input_units', 32, 256, step=64),
        activation=hp.Choice('input_activation', ['relu', 'tanh']),
        input_shape=(len(FEATURE_COLUMNS),),
        kernel_regularizer=l2(hp.Float('input_l2', 1e-5, 1e-3))
    ))

    # 隐藏层
    for i in range(num_layers):
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', 32, 512, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'swish', 'elu']),
            kernel_regularizer=l2(hp.Float(f'layer_{i}_l2', 1e-6, 1e-3))
        ))

        if hp.Boolean(f'layer_{i}_batch_norm'):
            model.add(BatchNormalization())

        model.add(Dropout(hp.Float(f'layer_{i}_dropout', 0.1, 0.5)))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 优化器配置
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=hp.Float('adam_beta1', 0.85, 0.99),
            beta_2=hp.Float('adam_beta2', 0.95, 0.9999)
        )
    else:
        optimizer = RMSprop(
            learning_rate=learning_rate,
            momentum=hp.Float('rmsprop_momentum', 0.8, 0.99)
        )

    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            AUC(name='auc'),  # 使用指标类实例
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    return model


def hyperparameter_tuning(X_train, y_train):
    """执行自动化超参数优化"""
    tuner = kt.BayesianOptimization(
        build_hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory='auto_nn',
        project_name='bp_tuning',
        overwrite=True
    )

    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    tuner.search(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    return tuner


def evaluate_model(model, X_test, y_test):
    """模型评估与可视化"""
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)

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
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')

    plt.subplot(1, 2, 2)
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title('性能指标')
    plt.savefig('performance.png')
    plt.close()

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    return metrics


# def method_project_split():
#     """项目划分方法（保持原有逻辑）"""
#     df = pd.read_excel(FILE_PATH)
#     projects = df['metrics'].unique()
#
#     all_train, all_test = pd.DataFrame(), pd.DataFrame()
#
#     for project in projects:
#         project_df = df[df['metrics'] == project]
#         X = project_df[FEATURE_COLUMNS]
#         y = project_df[LABEL_COLUMN]
#
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, stratify=y, random_state=42)
#
#         # 标准化处理
#         scaler = StandardScaler()
#         train_features = X_train[FEATURE_COLUMNS]
#         test_features = X_test[FEATURE_COLUMNS]
#
#         X_train[FEATURE_COLUMNS] = scaler.fit_transform(train_features)
#         X_test[FEATURE_COLUMNS] = scaler.transform(test_features)
#
#         all_train = pd.concat([all_train, pd.concat([X_train, y_train], axis=1)])
#         all_test = pd.concat([all_test, pd.concat([X_test, y_test], axis=1)])
#
#     # 保存结果
#     all_train.to_excel(r"C:\Users\17958\Desktop\project_train-1.xlsx", index=False)
#     all_test.to_excel(r"C:\Users\17958\Desktop\project_test-1.xlsx", index=False)


def main():
    # 数据预处理
    # method_project_split()

    try:
        # 加载数据
        X_train, X_test, y_train, y_test = load_preprocessed_data(
            r"C:\Users\17958\Desktop\project_train.xlsx",
            r"C:\Users\17958\Desktop\project_test.xlsx"
        )

        # print("\n=== 执行SMOTE过采样 ===")
        # # print("原始类别分布:", y_train.value_counts())
        #
        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        # print("过采样后分布:", pd.Series(y_train).value_counts())

        # 自动化参数搜索
        print("\n=== 开始自动化参数优化 ===")
        tuner = hyperparameter_tuning(X_train, y_train)
        best_model = tuner.get_best_models()[0]

        # 最终训练
        print("\n=== 最终模型训练 ===")
        history = best_model.fit(
            X_train, y_train,
            epochs=MAX_EPOCHS * 2,
            batch_size=128,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=20, monitor='val_auc', mode='max')],
            verbose=1
        )
        # 新增训练轨迹可视化 ------------------------
        print("\n=== 生成训练轨迹图 ===")
        plt.figure(figsize=(15, 6))

        # 损失变化曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('损失变化曲线', fontsize=12)
        plt.xlabel('训练轮次', fontsize=10)
        plt.ylabel('损失值', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        # AUC变化曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['auc'], label='训练AUC')
        plt.plot(history.history['val_auc'], label='验证AUC')
        plt.title('AUC变化曲线', fontsize=12)
        plt.xlabel('训练轮次', fontsize=10)
        plt.ylabel('AUC值', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300)
        plt.close()

        print("训练轨迹图已保存为 training_curves.png")
        # 模型评估
        print("\n=== 测试集评估结果 ===")
        metrics = evaluate_model(best_model, X_test, y_test)

        print("\n=== 最佳模型参数配置 ===")
        best_hps = tuner.get_best_hyperparameters()[0]

        # 获取所有超参数配置
        param_config = {
            '网络结构': {
                '输入层': {
                    'units': best_hps.get('input_units'),
                    'activation': best_hps.get('input_activation'),
                    'l2_reg': best_hps.get('input_l2')
                },
                '隐藏层': [],
                '输出层': {'activation': 'sigmoid'}
            },
            '优化器': {
                '类型': best_hps.get('optimizer'),
                '学习率': best_hps.get('lr')
            }
        }

        # 解析隐藏层参数
        num_layers = best_hps.get('num_layers')
        for i in range(num_layers):
            layer_params = {
                'units': best_hps.get(f'layer_{i}_units'),
                'activation': best_hps.get(f'layer_{i}_activation'),
                'l2_reg': best_hps.get(f'layer_{i}_l2'),
                'batch_norm': best_hps.get(f'layer_{i}_batch_norm'),
                'dropout_rate': best_hps.get(f'layer_{i}_dropout')
            }
            param_config['网络结构']['隐藏层'].append(layer_params)

        # 打印参数配置
        print("\n网络结构:")
        print(f"输入层: {param_config['网络结构']['输入层']}")
        for i, layer in enumerate(param_config['网络结构']['隐藏层']):
            print(f"隐藏层{i + 1}: {layer}")
        print(f"输出层: {param_config['网络结构']['输出层']}")

        print("\n优化器配置:")
        print(f"类型: {param_config['优化器']['类型']}")
        print(f"学习率: {param_config['优化器']['学习率']:.6f}")

        # 保存参数到文件
        with open('model_config.txt', 'w') as f:
            import json
            json.dump(param_config, f, indent=4, ensure_ascii=False)

        # 保存模型
        best_model.save('optimized_bp_model.h5')
        print("\n模型已保存为 optimized_bp_model.h5")
        print("参数配置已保存至 model_config.txt")

        # 保存模型
        best_model.save('optimized_bp_model.h5')
        print("\n模型已保存为 optimized_bp_model.h5")

    except Exception as e:
        print(f"运行异常: {str(e)}")


if __name__ == "__main__":
    main()