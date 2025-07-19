import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取Excel文件
file_path = r"C:\Users\17958\Desktop\train2.0_processed.xlsx"
df = pd.read_excel(file_path)

# 定义特征列
features = [
    "WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "B", "D", "E", "N",
    "n", "V", "CLOC", "NCLOC", "LOC", "Cyclic", "Dcy", "Dcy*", "DPT",
    "DPT*", "PDcy", "PDpt", "Command", "COM_RAT", "CONS", "MPC", "NAAC",
    "NAIC", "NOAC", "NOIC", "NOOC", "NTP", "Level", "Level*", "Inner",
    "INNER", "CSA", "CSO", "CSOA", "jf", "JM", "JLOC", "OCavg", "OCmax",
    "OPavg", "OSavg", "OSmax", "Query", "STAT", "SUB", "TCOM_RAT", "TODO"
]
target = '1适合LLM'

# 检查缺失值
print("缺失值统计：")
print(df[features + [target]].isnull().sum())

# 计算互信息
mi_scores = mutual_info_classif(df[features], df[target])

# 创建结果DataFrame
mi_df = pd.DataFrame({
    '特征': features,
    '互信息值': mi_scores
}).sort_values('互信息值', ascending=False)

# 可视化设置
plt.figure(figsize=(12, 8))
bars = plt.barh(mi_df['特征'], mi_df['互信息值'], color='#2c7fb8')

# 添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}',
             va='center', ha='left', fontsize=10)

plt.xlabel('互信息值', fontsize=12)
plt.title('特征与目标变量的非线性相关性', fontsize=14, pad=20)
plt.gca().invert_yaxis()  # 倒序显示
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 保存高清图像
plt.savefig('mutual_info_plot.png', dpi=300, bbox_inches='tight')
print("可视化结果已保存为 mutual_info_plot.png")

# 显示结果
print("\n互信息分析结果：")
print(mi_df)