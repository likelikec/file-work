import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ----------------------------------
# 公共配置部分
# ----------------------------------
file_path = r"C:\Users\17958\Desktop\symtrain.xlsx"
feature_columns = ['mio','B', 'COM_RAT', 'Cyclic', 'D',
            'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
            'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT',
            'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
            'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n',
            'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
            'Query', 'RFC', 'TODO', "String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling",'class']
label_column = "1适合LLM"


def preprocess_data(df):
    """数据预处理：处理缺失值和数据类型"""
    # 检查缺失值
    print(f"原始数据缺失值数量:\n{df.isnull().sum()}")

    # 方法1：删除含有缺失值的行（适用于少量缺失）
    # df = df.dropna(subset=feature_columns + [label_column])

    # 方法2：用中位数填充数值特征（推荐）
    imputer = SimpleImputer(strategy='median')
    df[feature_columns] = imputer.fit_transform(df[feature_columns])

    # 确保标签列无缺失
    if df[label_column].isnull().any():
        df = df.dropna(subset=[label_column])

    print(f"处理后的缺失值数量:\n{df.isnull().sum()}")
    return df


# ----------------------------------
# 方法：分层划分 (Stratified Split)
# ----------------------------------
# def method_stratified():
#     # 读取并预处理数据
#     df = pd.read_excel(file_path)
#
#
#     X, y = df[feature_columns], df[label_column]
#
#     # 分层划分8-2
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         stratify=y,
#         random_state=42
#     )
#
#     # 保存结果
#     pd.concat([X_train, y_train], axis=1).to_excel(
#         r"C:\Users\17958\Desktop\strat_train.xlsx", index=False
#     )
#     pd.concat([X_test, y_test], axis=1).to_excel(
#         r"C:\Users\17958\Desktop\strat_test.xlsx", index=False
#     )
# # ----------------------------------
# 方法：随机划分 (Random Split)
# ----------------------------------
def method_random():
    # 读取并预处理数据
    df = pd.read_excel(file_path)

    X, y = df[feature_columns], df[label_column]

    # 随机划分8-2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 保存结果
    pd.concat([X_train, y_train], axis=1).to_excel(
        r"C:\Users\17958\Desktop\sym-random_train.xlsx", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_excel(
        r"C:\Users\17958\Desktop\sym-random_test.xlsx", index=False
    )


def method_project_split():
    # 读取并预处理数据
    df = pd.read_excel(file_path)
    # df = preprocess_data(df)
    df[feature_columns] = df[feature_columns].fillna(-1)
    # 获取所有项目名称
    projects = df['metrics'].unique()
    print(f"发现 {len(projects)} 个项目: {projects}")

    # 初始化存储容器
    all_train = pd.DataFrame()
    all_test = pd.DataFrame()

    # 遍历每个项目进行划分
    for project in projects:
        # 提取当前项目数据
        project_df = df[df['metrics'] == project]

        # 分层划分8-2
        X = project_df[feature_columns]
        y = project_df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 合并当前项目的划分结果
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        all_train = pd.concat([all_train, train_df])
        all_test = pd.concat([all_test, test_df])

    # 验证数据完整性
    print(f"\n总训练集样本数: {len(all_train)} | 总测试集样本数: {len(all_test)}")
    print(f"原始数据总样本数: {len(df)} | 合并后总数: {len(all_train) + len(all_test)}")

    # 保存结果
    all_train.to_excel(r"C:\Users\17958\Desktop\sym-train01.xlsx", index=False)
    all_test.to_excel(r"C:\Users\17958\Desktop\sym-test01.xlsx", index=False)
# ----------------------------------
# 执行方法
# ----------------------------------
if __name__ == "__main__":
    # 先验证原始数据
    raw_df = pd.read_excel(file_path)
    print("\n原始数据统计:")
    print(f"总样本数: {len(raw_df)}")
    print(f"类别分布:\n{raw_df[label_column].value_counts()}")

    # method_stratified()
    # method_random()
    method_project_split()