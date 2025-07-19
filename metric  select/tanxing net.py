import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# 定义文件路径和列名
FILE_PATH = r"C:\Users\17958\Desktop\train_4.0.xlsx"
FEATURES = ['B', 'COM_RAT', 'Cyclic', 'D',
            'Dcy*', 'DIT', 'DPT*', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
            'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'SUB',
            'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
            'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'MPC', 'n',
            'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
            'Query', 'RFC', 'TODO']
TARGET = "1适合LLM"

try:
    # 读取数据
    df = pd.read_excel(FILE_PATH)

    # ------------------ 新增部分：处理缺失值 ------------------
    # 删除特征列和目标列的缺失值
    df = df.dropna(subset=FEATURES + [TARGET])

    # 检查数据是否存在（处理缺失值后）
    if df.empty:
        raise ValueError("删除缺失值后数据为空，请检查数据质量")
    # --------------------------------------------------------

    # 分离特征和标签
    X = df[FEATURES]
    y = df[TARGET]

    # 数据标准化（正则化需要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 设置弹性网络参数
    l1_ratios = np.linspace(0.1, 0.9, 5)  # 从0.1到0.9取5个值
    Cs = 10  # 正则化强度候选值数量

    # 创建模型（使用saga求解器支持弹性网络）
    model = LogisticRegressionCV(
        penalty='elasticnet',
        solver='saga',
        Cs=Cs,
        cv=5,
        l1_ratios=l1_ratios,
        max_iter=10000,
        class_weight='balanced',  # 处理类别不平衡
        random_state=42
    )

    # 训练模型
    model.fit(X_scaled, y)

    # 获取非零系数特征
    selected_features = X.columns[np.abs(model.coef_[0]) > 1e-6].tolist()

    # 输出结果
    print(f"筛选出 {len(selected_features)} 个重要特征：")
    print("\n".join(selected_features))

    # 输出模型评估
    print(f"\n最佳正则化强度(C)：{model.C_[0]:.4f}")
    print(f"最佳L1比例：{model.l1_ratio_[0]:.2f}")

except FileNotFoundError:
    print(f"文件 {FILE_PATH} 未找到，请检查路径是否正确")
except KeyError as e:
    print(f"列名错误：{str(e)} 不存在于数据中")
except Exception as e:
    print(f"运行时错误：{str(e)}")