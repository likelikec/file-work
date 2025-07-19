import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\testarttrain.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 定义所有分析变量
all_vars =FEATURE_COLUMNS=['1适合LLM','B', 'COM_RAT', 'Cyclic', 'D',
            'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
            'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT',
            'TCOM_RAT', 'V', 'WMC', 'CBO', 'CLOC', 'Command', 'CONS', 'CSA', 'CSO',
            'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l', 'MPC', 'n',
            'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax',
            'Query', 'RFC', 'TODO', "String processing", "File operations", "Network communication",
            "Database operations", "Mathematical calculation", "User Interface",
            "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
            "Concurrency and Multithreading", "Exception handling"]
# 检查数据列是否存在
missing_cols = [col for col in all_vars if col not in df.columns]
if missing_cols:
    print(f"以下列名在文件中未找到：{missing_cols}")
    exit()

# 计算完整的Spearman相关性矩阵
corr_matrix = df[all_vars].corr(method='spearman')

# 保存完整相关性矩阵到Excel
corr_matrix.to_excel("full_spearman_correlation.xlsx")

print("完整相关性矩阵已保存至 full_spearman_correlation.xlsx")

# 可视化设置
plt.figure(figsize=(24, 20))

# 创建热力图
sns.heatmap(corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Spearman 相关系数'})

# 美化图表
plt.title('完整变量间 Spearman 相关性热力图', fontsize=24, pad=25)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()

# 保存高清图片
plt.savefig('full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("热力图已保存至 full_correlation_heatmap.png")

# 生成详细的相关性报告（包含p值）
correlation_report = []

for i in range(len(all_vars)):
    for j in range(i+1, len(all_vars)):
        var1 = all_vars[i]
        var2 = all_vars[j]
        valid_data = df[[var1, var2]].dropna()
        if len(valid_data) > 1:
            corr, p_value = spearmanr(valid_data[var1], valid_data[var2])
            correlation_report.append({
                'Variable1': var1,
                'Variable2': var2,
                'Correlation': corr,
                'P-value': p_value
            })
        else:
            correlation_report.append({
                'Variable1': var1,
                'Variable2': var2,
                'Correlation': None,
                'P-value': None
            })

# 转换为DataFrame并保存
report_df = pd.DataFrame(correlation_report)
report_df.to_excel("detailed_correlation_report.xlsx", index=False)

print("详细相关性报告已保存至 detailed_correlation_report.xlsx")