import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 文件路径
file_path = r"C:\Users\17958\Desktop\train-1.0.xlsx"

# 读取Excel文件
df = pd.read_excel(file_path)

# 定义因变量和自变量
dependent_var = '1适合LLM'
independent_vars = ["Cyclic", "Dcy", "Dcy*", "Dpt", "Dpt*", "PDcy", "PDpt", "OCavg", "OCmax", "WMC", "CLOC", "JLOC", "LOC"]

# 设置图形布局：4行4列
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
fig.suptitle('Boxplots of Independent Variables by 1适合LLM', fontsize=16)

# 展平axes数组以便于索引
axes = axes.flatten()

# 为每个自变量绘制箱线图
for i, var in enumerate(independent_vars):
    if i < len(axes):  # 确保不超出子图数量
        sns.boxplot(x=dependent_var, y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var}')
        axes[i].set_xlabel(dependent_var)
        axes[i].set_ylabel(var)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# 调整布局以防止重叠
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存图形为PNG文件
plt.savefig(r"C:\Users\17958\Desktop\boxplots_combined.png", dpi=300)

# 显示图形（可选）
plt.show()