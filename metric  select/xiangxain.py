import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 文件路径
file_path = r"C:\Users\17958\Desktop\train2.0_processed.xlsx"

# 读取Excel文件
df = pd.read_excel(file_path)

# 定义因变量和自变量
dependent_var = '1适合LLM'
independent_vars = [
    "WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "B", "D", "E", "N",
    "n", "V", "CLOC", "NCLOC", "LOC", "Cyclic", "Dcy", "Dcy*", "DPT",
    "DPT*", "PDcy", "PDpt", "Command", "COM_RAT", "CONS", "MPC", "NAAC",
    "NAIC", "NOAC", "NOIC", "NOOC", "NTP", "Level", "Level*", "Inner",
    "INNER", "CSA", "CSO", "CSOA", "jf", "JM", "JLOC", "OCavg", "OCmax",
    "OPavg", "OSavg", "OSmax", "Query", "STAT", "SUB", "TCOM_RAT", "TODO"
]

# 将52个指标分成4组，每组13个
groups = [independent_vars[i * 13:(i + 1) * 13] for i in range(4)]

# 设置通用子图布局参数
ROWS_PER_PLOT = 3
COLS_PER_PLOT = 5

# 为每组创建单独的图表
for group_num, group_vars in enumerate(groups):
    # 创建新图表
    fig, axes = plt.subplots(ROWS_PER_PLOT, COLS_PER_PLOT,
                             figsize=(20, 12))
    fig.suptitle(f'指标分布分析 - 第 {group_num + 1} 组',
                 fontsize=16, y=0.98)

    # 展平子图数组
    axes = axes.flatten()

    # 绘制每个指标的箱线图
    for i, var in enumerate(group_vars):
        sns.boxplot(x=dependent_var, y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var}', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # 添加全局标签
    fig.text(0.5, 0.04, dependent_var, ha='center', va='center')
    fig.text(0.05, 0.5, '指标值', ha='center', va='center',
             rotation='vertical')

    # 调整布局
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])

    # 保存图表
    plt.savefig(
        fr"C:\Users\17958\Desktop\boxplot_group_{group_num + 1}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

print("图表生成完成，已保存到桌面")