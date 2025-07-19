import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

# 数据配置
categories = ['Cli', 'Csv', 'Gson', 'Lang']

# 颜色配置
colors = {
    'line': ['#D8BFD8', "#D0B1D9","#C8A4D9", "#A37CDB", '#9370DB', '#6A5ACD', '#4B0082'],
    'branch': ['#E0FFFF', "#D0F7F9", "#C0EFF3", '#87CEEB', '#6495ED', '#1E90FF']
}

# 全局字体设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'bold',  # 全局加粗
    'axes.labelsize': 34,
    'axes.titlesize': 22,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=300)


def plot_bars(ax, data, color_palette, title, edge_color):
    x = np.arange(len(categories))
    bar_width = 0.18
    spacing = 0.03

    positions = [x - 1.5 * bar_width - 1.5 * spacing,
                 x - 0.5 * bar_width - 0.5 * spacing,
                 x + 0.5 * bar_width + 0.5 * spacing,
                 x + 1.5 * bar_width + 1.5 * spacing]

    # 绘制柱状图
    for idx, (method, values) in enumerate(data.items()):
        bars = ax.bar(positions[idx], values, bar_width,
                      color=color_palette[idx],
                      edgecolor=edge_color,
                      linewidth=0.8,
                      label=method)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=14, color='black',
                    fontweight='bold')

    # 坐标轴设置
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories,
                       rotation=0,  # 修改为水平显示
                       ha='center',  # 水平居中
                       fontweight='bold',
                       fontsize=22)  # 可根据需要调整字号

    # 纵坐标设置
    ax.set_ylim(0, 1.0)

    # 刻度定位器
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # 刻度格式器
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))

    # 刻度参数设置
    ax.tick_params(axis='y', which='major',
                   length=8,
                   width=1.2,
                   color='black',
                   labelsize=18,
                   labelcolor='black',
                   labelrotation=0,
                   pad=8)

    ax.tick_params(axis='y', which='minor',
                   length=5,
                   width=1.0,
                   color='black')

    # 边框设置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.legend(loc='upper left', frameon=False, fontsize=10)


# 生成图表
plot_bars(ax1,
          {'ChatGPT-3.5': [0.44, 0.51, 0.40, 0.41],
           'ChatUniTest': [0.58, 0.53, 0.46, 0.62],
           'ChatGPT-4.0': [0.59, 0.64, 0.60, 0.57],
           'SUMMIT': [0.91, 0.70, 0.66, 0.80]},
          colors['line'],
          "Line Coverage of Correct Test",
          edge_color='#4B0082')

plot_bars(ax2,
          {'ChatGPT-3.5': [0.28, 0.31, 0.29, 0.33],
           'ChatUniTest': [0.56, 0.36, 0.45, 0.57],
           'ChatGPT-4.0': [0.52, 0.42, 0.47, 0.56],
           'SUMMIT': [0.92, 0.74, 0.66, 0.83]},
          colors['branch'],
          "Branch Coverage of Correct Test",
          edge_color='#191970')

plt.tight_layout(pad=5, w_pad=6, h_pad=4)
plt.savefig('final_plot.pdf', bbox_inches='tight', dpi=300)
plt.show()