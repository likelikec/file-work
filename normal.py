import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
import os

# 配置参数
FILE_PATH = r"C:\Users\17958\Desktop\train.xlsx"
COLUMNS_TO_CHECK = [
    'Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*',
    'PDcy', 'PDpt', 'OCavg', 'OCmax', 'WMC',
    'CLOC', 'JLOC', 'LOC'
]
OUTPUT_EXCEL = r"C:\Users\17958\Desktop\normality_test_results.xlsx"


def load_data(file_path):
    """加载Excel数据"""
    df = pd.read_excel(file_path)
    missing_cols = [col for col in COLUMNS_TO_CHECK if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列不存在于数据集中: {missing_cols}")
    return df[COLUMNS_TO_CHECK]


def shapiro_test(df):
    """执行Shapiro-Wilk检验并计算统计量"""
    results = []

    for col in df.columns:
        data = df[col].dropna()
        n = len(data)

        # 基础统计量
        stat, p = stats.shapiro(data)
        median = np.median(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # 样本标准差

        # 分布形态
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)  # 超额峰度

        results.append({
            'Column': col,
            'Sample Size': n,
            'Median': round(median, 4),
            'Mean': round(mean, 4),
            'Std Dev': round(std, 4),
            'Skewness': round(skewness, 4),
            'Kurtosis': round(kurt, 4),
            'Shapiro Stat': round(stat, 4),
            'p-value': round(p, 4),
            'Normality': 'Normal' if p > 0.05 else 'Not Normal'
        })

    return pd.DataFrame(results)


def plot_ppplots(df):
    """绘制集成P-P图"""
    plt.figure(figsize=(15, 15))

    # 在主图上添加标题（调整y参数控制位置）
    plt.suptitle("P-P Plots for All Features",
                 y=0.05,  # 控制标题位置（0-1，0为底部）
                 fontsize=14,
                 verticalalignment='bottom')  # 垂直对齐方式

    for i, col in enumerate(df.columns, 1):
        plt.subplot(4, 4, i)
        data = df[col].dropna()

        probplot = ProbPlot(data, fit=True)
        osm = probplot.theoretical_quantiles
        osr = probplot.sample_quantiles

        plt.scatter(osm, osr, s=10, alpha=0.5)
        plt.plot(np.linspace(osm.min(), osm.max()),
                 np.linspace(osm.min(), osm.max()),
                 'r--', lw=1)
        plt.title(f"{col}\n(n={len(data)})", fontsize=8)
        plt.xlabel('')
        plt.ylabel('')

    # 调整布局防止标题被遮挡
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 增加底部边距

    plt.savefig(os.path.join(os.path.dirname(OUTPUT_EXCEL), 'combined_pp_plots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    df = load_data(FILE_PATH)
    results_df = shapiro_test(df)

    # 调整列顺序
    columns_order = [
        'Column', 'Sample Size', 'Mean', 'Median', 'Std Dev',
        'Skewness', 'Kurtosis', 'Shapiro Stat', 'p-value', 'Normality'
    ]
    results_df = results_df[columns_order]

    results_df.to_excel(OUTPUT_EXCEL, index=False)
    plot_ppplots(df)
    print(f"结果已保存至：{OUTPUT_EXCEL}\nP-P图已保存至桌面")


if __name__ == "__main__":
    main()