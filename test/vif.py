import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_vif(df, variables):
    """
    计算数据框中指定变量的方差膨胀因子(VIF)
    """
    # 创建一个数据框来存储VIF结果
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables

    # 确保数据没有缺失值
    X = df[variables].dropna()

    # 标准化数据（VIF对变量尺度敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 计算每个变量的VIF
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

    # 按VIF降序排序
    return vif_data.sort_values(by="VIF", ascending=False)


def check_data_distribution(df, variables, sample_size=1000):
    """
    检查变量的数据分布（正态性检验）
    """
    # 由于Shapiro-Wilk检验对大样本过于敏感，随机抽取一部分数据进行检验
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    # 存储检验结果
    normality_results = []

    for var in variables:
        if var in df_sample.columns:
            # 进行Shapiro-Wilk正态性检验
            stat, p = stats.shapiro(df_sample[var].dropna())
            normality_results.append({
                '变量': var,
                '统计量': stat,
                'p值': p,
                '是否正态(α=0.05)': p > 0.05
            })

    return pd.DataFrame(normality_results)


def identify_high_correlations(df, variables, threshold=0.8, method='spearman'):
    """
    识别相关系数绝对值超过阈值的变量对
    method: 'pearson'、'spearman' 或 'kendall'
    """
    # 计算相关系数矩阵
    corr_matrix = df[variables].dropna().corr(method=method)

    # 创建一个空列表来存储高相关变量对
    high_correlations = []

    # 遍历相关系数矩阵的上三角部分（避免重复）
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_correlations.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    # 按相关系数绝对值排序
    high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    return high_correlations, corr_matrix


def visualize_correlation_matrix(corr_matrix, title="相关系数矩阵"):
    """
    可视化相关系数矩阵
    """
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 上三角掩码
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # 蓝红配色方案

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()


def visualize_vif(vif_data, title="方差膨胀因子分析"):
    """
    可视化VIF结果
    """
    plt.figure(figsize=(12, 8))

    # 按VIF值降序排序
    vif_data_sorted = vif_data.sort_values(by="VIF", ascending=True)

    # 绘制条形图
    bars = plt.barh(vif_data_sorted['Variable'], vif_data_sorted['VIF'])

    # 添加阈值线
    plt.axvline(x=5, color='orange', linestyle='--', label='中度共线性阈值 (VIF=5)')
    plt.axvline(x=10, color='red', linestyle='--', label='严重共线性阈值 (VIF=10)')

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2f}', ha='left', va='center', fontsize=9)

    plt.xlabel('方差膨胀因子 (VIF)')
    plt.ylabel('变量')
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('vif_analysis.png', dpi=300)
    plt.show()


def main():
    # 文件路径
    file_path = r"C:\Users\17958\Desktop\symtrain.xlsx"

    # 读取 Excel 文件
    try:
        df = pd.read_excel(file_path)
        print(f"数据加载成功，共 {df.shape[0]} 行，{df.shape[1]} 列")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 定义自变量列名（根据实际数据修改）
    independent_vars = ['B', 'COM_RAT', 'Cyclic',
            'Dc+y', 'DIT', 'DP+T',  'Inner', 'LCOM', 'Level',
             'NOAC', 'NOC',  'PDcy', 'PDpt',
                'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
             'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l',
              'NOOC', 'NTP', 'OCavg', 'OPavg',
              'TODO', "String processing", "File operations",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]

    # 检查列名是否存在
    missing_cols = [col for col in independent_vars if col not in df.columns]
    if missing_cols:
        print(f"以下列名在文件中未找到：{missing_cols}")
        print("请检查自变量列表是否正确！")
        return

    # 数据预览
    print("\n数据前几行预览：")
    print(df.head().to_string())

    # 0. 检查数据分布
    print("\n=== 数据分布检查 ===")
    normality_results = check_data_distribution(df, independent_vars)
    print("\n数据正态性检验结果（Shapiro-Wilk检验）：")
    print(normality_results)

    # 计算正态分布变量的比例
    normal_vars_ratio = normality_results['是否正态(α=0.05)'].mean() * 100
    print(f"\n正态分布变量比例：{normal_vars_ratio:.1f}%")

    # 根据正态性检验结果选择相关系数方法
    if normal_vars_ratio > 70:
        corr_method = 'pearson'
        print("\n提示：多数变量符合正态分布，将使用皮尔逊相关系数。")
    else:
        corr_method = 'spearman'
        print("\n提示：多数变量不满足正态分布，将使用斯皮尔曼等级相关系数。")

    # 1. 计算方差膨胀因子(VIF)
    print("\n=== 计算方差膨胀因子(VIF) ===")
    vif_data = calculate_vif(df, independent_vars)

    print("\n方差膨胀因子结果：")
    print(vif_data)

    # 2. 识别高度相关的变量对
    print("\n=== 识别高度相关的变量对 ===")
    threshold = 0.8  # 相关系数阈值
    high_correlations, corr_matrix = identify_high_correlations(df, independent_vars, threshold, corr_method)

    print(f"\n高度相关的变量对（相关系数 > {threshold}，方法：{corr_method}）：")
    if high_correlations:
        for var1, var2, corr in high_correlations:
            print(f"{var1} 和 {var2}: 相关系数 = {corr:.4f}")
    else:
        print("未发现高度相关的变量对")

    # 3. 可视化分析
    print("\n=== 可视化分析 ===")

    # 可视化相关系数矩阵
    visualize_correlation_matrix(corr_matrix, f"自变量相关系数矩阵（方法：{corr_method}）")

    # 可视化VIF结果
    visualize_vif(vif_data, "自变量方差膨胀因子分析")

    # 4. 保存结果
    vif_data.to_excel("collinearity_vif_results.xlsx", index=False)
    if high_correlations:
        high_corr_df = pd.DataFrame(high_correlations, columns=['变量1', '变量2', '相关系数'])
        high_corr_df.to_excel(f"high_correlations_{corr_method}.xlsx", index=False)

    print("\n=== 分析完成 ===")
    print(f"方差膨胀因子结果已保存至 'collinearity_vif_results.xlsx'")
    if high_correlations:
        print(f"高度相关变量对已保存至 'high_correlations_{corr_method}.xlsx'")
    print("相关系数矩阵图表已保存至 'correlation_matrix.png'")
    print("VIF分析图表已保存至 'vif_analysis.png'")


if __name__ == "__main__":
    main()