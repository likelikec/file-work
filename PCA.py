import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取Excel文件
file_path = r'C:\Users\17958\Desktop\类覆盖率+指标.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 列选择（O到AC列，对应Excel列号15-29）
start_col = 14  # O列是第15列（索引从0开始为14）
end_col = 29  # AC列是第29列（索引28）
target_columns = df.iloc[:, start_col:end_col + 1]

# 数据预处理
print("原始数据形状:", target_columns.shape)

# 处理缺失值（删除包含缺失值的行）
clean_data = target_columns.dropna()
print("清洗后数据形状:", clean_data.shape)

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_data)

# 执行PCA（保留95%方差）
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(scaled_data)


# 结果解析
# 结果解析
def print_pca_results(pca):
    print("\n=== PCA分析结果 ===")
    print(f"保留主成分数: {pca.n_components_}")
    print("各主成分解释方差比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i + 1}: {ratio:.2%}")

    print("\n累积解释方差:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, total in enumerate(cumulative):
        print(f"前{i + 1}个主成分: {total:.2%}")


# 生成结果DataFrame
pca_df = pd.DataFrame(
    pca_result,
    columns=[f'PC{i + 1}' for i in range(pca.n_components_)]
)

# 主成分载荷矩阵
loadings = pd.DataFrame(
    pca.components_.T,
    index=clean_data.columns,
    columns=pca_df.columns
)

# 输出结果
print_pca_results(pca)

# 修改后的载荷矩阵显示
print("\n主成分载荷矩阵:")
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(loadings.round(3))
pd.reset_option('display.max_rows')
pd.reset_option('display.float_format')

# 将结果合并回原始数据（可选）
final_df = pd.concat([df.iloc[clean_data.index], pca_df], axis=1)

# 保存结果到新Excel文件（可选）
final_df.to_excel(r'C:\Users\17958\Desktop\PCA分析结果.xlsx', index=False)
print("\n结果已保存到桌面PCA分析结果.xlsx")