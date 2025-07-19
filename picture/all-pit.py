import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

def load_and_filter_data(file_path):
    """加载并筛选 P-value ≤ 0.01 的变量"""
    df = pd.read_excel(file_path)
    # 提取 95% CI 上下限（若不需要可删除）
    df[['CI_lower', 'CI_upper']] = df['95% CI (OR)'].str.extract(r'\[(\d+\.\d+),\s*(\d+\.\d+)\]').astype(float)
    # 筛选 + 按 OR 值排序
    return df[df['P-value'] <= 0.01].sort_values('Odds Ratio')

def plot_bubble_plot(df, output_path=None):
    plt.figure(figsize=(12, 9))  # 适度缩小画布避免过度留白

    # 气泡大小：与 Z 值绝对值成正比（增大基础系数让气泡更明显）
    z_abs = df['Z Value'].abs()
    sizes = 300 * (z_abs / z_abs.max())

    # 颜色映射：OR>1 红、OR<1 蓝，深浅体现显著性
    colors = []
    for or_val, p_val in zip(df['Odds Ratio'], df['P-value']):
        alpha = 1 - min(p_val * 100, 0.9)  # P 越小，颜色越深
        colors.append((1, 1-alpha, 1-alpha) if or_val > 1 else (1-alpha, 1-alpha, 1))

    # 绘制气泡
    scatter = plt.scatter(
        df['Odds Ratio'], df['Variable'],
        s=sizes, c=colors, edgecolors='black', alpha=0.7
    )

    # 参考线（OR=1）
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)

    # 标题与坐标轴
    plt.title('气泡图：单变量逻辑回归结果（P≤0.01）', fontsize=15)
    plt.xlabel('比值比 (OR)', fontsize=12)
    plt.ylabel('变量', fontsize=12)
    plt.yticks(fontsize=10)  # 缩小 y 轴字体，给气泡更多空间

    # 关键优化：动态计算文本位置，让文字“贴紧”气泡
    for i, (or_val, sig, size) in enumerate(zip(df['Odds Ratio'], df['Significance'], sizes)):
        # 1. 计算气泡实际半径（matplotlib 中 s = πr² → r = √(s/π)）
        bubble_radius = (size / np.pi) ** 0.5
        # 2. 动态偏移：半径越大，偏移越小（让文字贴紧气泡）
        text_offset = bubble_radius / 500
        # 3. 最终 X 坐标 = 气泡右侧边缘 + 微小间距
        x_pos = or_val + bubble_radius / 75 + text_offset

        # 格式化文本：强制分行显示
        text = f"OR值={or_val:.2f}\nSignificance值={sig}"
        # 背景半透明，避免遮挡
        plt.text(
            x_pos, i, text,
            ha='left', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )

    # 优化 X 轴范围：仅保留“气泡最右 + 必要文字空间”，避免过度留白
    max_or = df['Odds Ratio'].max()
    max_text_width = 0.2 * max_or  # 文字最大宽度（可微调）
    plt.xlim(
        df['Odds Ratio'].min() * 0.95,  # 左侧留 5% 空间
        max_or + max_text_width        # 右侧仅保留文字所需空间
    )

    # 调整布局（避免标题、标签挤压）
    plt.tight_layout()

    # 保存（可选）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return plt

if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = r"D:\project_py\file-work\test\logistic-testart_bootstrap_converged.xlsx"
    filtered_df = load_and_filter_data(file_path)
    print(f"筛选出 {len(filtered_df)} 个 P-value ≤ 0.01 的变量")
    # 绘图 + 显示
    plot_bubble_plot(filtered_df).show()