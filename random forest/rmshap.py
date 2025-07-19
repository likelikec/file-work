import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "Microsoft JhengHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件路径
model_path = r'C:\Users\17958\Desktop\01random_forest_model.pkl'
test_file = r'C:\Users\17958\Desktop\01_test.xlsx'
output_shap_values = r'C:\Users\17958\Desktop\01SHAP值.xlsx'
output_shap_plot = r'C:\Users\17958\Desktop\01SHAP值分布.png'
output_waterfall_plot = r'C:\Users\17958\Desktop\01SHAP瀑布图.png'

# 定义特征列
feature_columns = ['B', 'COM_RAT', 'Cyclic', 'D', 'Dc+y', 'DIT', 'DP+T', 'E', 'Inner', 'LCOM', 'Level', 'LOC', 'N',
                   'NCLOC', 'NOAC', 'NOC', 'NOIC', 'OCmax', 'PDcy', 'PDpt', 'STAT', 'TCOM_RAT', 'V', 'WMC', 'CBO',
                   'CLOC', 'Command', 'CONS', 'CSA', 'CSO', 'CSOA', 'Dcy', 'DPT', 'INNER', 'jf', 'JLOC', 'Jm', 'Leve+l',
                   'MPC', 'n', 'NAAC', 'NAIC', 'NOOC', 'NTP', 'OCavg', 'OPavg', 'OSavg', 'OSmax', 'Query', 'RFC',
                   'TODO',
                   "String processing", "File operations", "Network communication", "Database operations",
                   "Mathematical calculation",
                   "User Interface", "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
                   "Concurrency and Multithreading", "Exception handling"]

# 加载模型
try:
    model = joblib.load(model_path)
    print("模型加载成功")
except Exception as e:
    print(f"加载模型出错: {e}")
    exit()

# 加载测试数据
try:
    test_data = pd.read_excel(test_file)
    X_test = test_data[feature_columns]
    print(f"测试数据形状: {X_test.shape}")
except Exception as e:
    print(f"读取测试数据出错: {e}")
    exit()

# 计算SHAP值
try:
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值 (可能需要根据数据量调整样本大小)
    if len(X_test) > 1000:
        shap_values = explainer.shap_values(X_test.sample(1000, random_state=42))
        print("由于数据量大，使用了1000个样本计算SHAP值")
    else:
        shap_values = explainer.shap_values(X_test)

    # 如果是二分类问题，我们只取正类的SHAP值
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    print("SHAP值计算完成")
except Exception as e:
    print(f"计算SHAP值出错: {e}")
    exit()

# 创建SHAP值的DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_columns)

# 计算特征重要性（SHAP值的绝对值平均）
shap_importance = pd.DataFrame({
    'Feature': feature_columns,
    'SHAP Importance': np.abs(shap_values).mean(axis=0)
})

# 计算SHAP值的平均值（代表特征对预测的平均影响方向和大小）
shap_mean = pd.DataFrame({
    'Feature': feature_columns,
    'SHAP Mean Value': shap_values.mean(axis=0)  # 新增：计算SHAP值的平均值
})

# 按SHAP重要性排序
shap_importance = shap_importance.sort_values('SHAP Importance', ascending=False)
sorted_features = shap_importance['Feature'].tolist()

# 按SHAP平均值的绝对值排序（便于分析主要影响特征）
shap_mean['SHAP Mean Abs'] = shap_mean['SHAP Mean Value'].abs()
shap_mean = shap_mean.sort_values('SHAP Mean Abs', ascending=False)

# 保存SHAP值到Excel
try:
    # 创建一个ExcelWriter对象
    with pd.ExcelWriter(output_shap_values) as writer:
        # 保存原始SHAP值
        shap_df.to_excel(writer, sheet_name='SHAP原始值', index=False)
        # 保存SHAP重要性
        shap_importance.to_excel(writer, sheet_name='SHAP重要性', index=False)
        # 新增：保存SHAP值的平均值
        shap_mean.to_excel(writer, sheet_name='SHAP平均值', index=False)
    print(f"SHAP值已保存到 {output_shap_values}")
except Exception as e:
    print(f"保存SHAP值出错: {e}")

# 创建Explanation对象
explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test.values,
    feature_names=feature_columns
)

# 创建SHAP值分布的summary plot（使用Explanation对象）
try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation, plot_type="bar", max_display=len(feature_columns), show=False)
    plt.tight_layout()
    plt.savefig(output_shap_plot, dpi=300, bbox_inches='tight')
    print(f"SHAP重要性条形图已保存到 {output_shap_plot}")

    # 创建SHAP值分布的蜂群图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation, max_display=len(feature_columns), show=False)
    plt.tight_layout()
    output_shap_beeswarm = output_shap_plot.replace('.png', '_蜂群图.png')
    plt.savefig(output_shap_beeswarm, dpi=300, bbox_inches='tight')
    print(f"SHAP值蜂群图已保存到 {output_shap_beeswarm}")

    # 创建SHAP瀑布图（以第一个样本为例）
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation[0], max_display=60, show=False)
    plt.tight_layout()
    plt.savefig(output_waterfall_plot, dpi=300, bbox_inches='tight')
    print(f"SHAP瀑布图已保存到 {output_waterfall_plot}")

    plt.close('all')
except Exception as e:
    print(f"生成SHAP图出错: {e}")