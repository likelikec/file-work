import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def collinearity_analysis(X):
    """计算条件数判断共线性程度"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 主成分分析提取特征值
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_

    # 计算条件数
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)

    if min_eig <= 1e-10:
        cond_number = np.inf
    else:
        cond_number = np.sqrt(max_eig / min_eig)

    # 判断共线性程度
    if cond_number < 10:
        interpretation = "共线性极弱"
    elif 10 <= cond_number < 30:
        interpretation = "共线性可接受"
    else:
        interpretation = "存在有害共线性（条件数 > 30）"

    return cond_number, interpretation


# 读取数据并计算条件数
if __name__ == "__main__":
    # 读取Excel文件
    file_path = r"C:\Users\17958\Desktop\symtrain.xlsx"
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件失败：{e}")
        exit()

    # 提取自变量列（确保所有变量都在数据中）
    independent_vars = [ 'COM_RAT', 'Cyclic',
            'Dc+y', 'DIT', 'DP+T',  'Inner', 'LCOM', 'Level',  'NOC', 'NOIC', 'OCmax',  'PDpt',
              'WMC',  'CLOC', 'Command', 'CONS',
             'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l',  'n',
            'NAAC',  'NOOC', 'NTP', 'OCavg', 'OPavg',
              'TODO', "String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]

    # 检查并筛选存在的变量
    existing_vars = [var for var in independent_vars if var in df.columns]
    missing_vars = [var for var in independent_vars if var not in df.columns]
    if missing_vars:
        print(f"警告：数据中缺少以下自变量，已自动忽略：{missing_vars}")

    if not existing_vars:
        print("错误：没有找到任何自变量列，请检查变量名是否正确。")
        exit()

    # 提取自变量数据并删除缺失值
    X = df[existing_vars].dropna().values
    if X.shape[0] == 0:
        print("错误：自变量数据中存在大量缺失值，无法进行计算。")
        exit()

    # 计算条件数
    cond_num, info = collinearity_analysis(X)
    print(f"\n自变量数量：{X.shape[1]}，有效样本数：{X.shape[0]}")
    print(f"条件数: {cond_num:.2f}")
    print(f"共线性判断: {info}")