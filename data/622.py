import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 读取Excel文件（注意路径中的反斜杠需要转义或使用原始字符串）
file_path = r"C:\Users\17958\Desktop\total.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')

# 2. 定义需要保留的非数值列（类名和分类标签）
non_numeric_cols = ["testart","metrics", "1适合LLM"]  # 替换为你的实际列名！
numeric_cols = data.columns.drop(non_numeric_cols)  # 自动排除非数值列

# 3. 标准化数值列
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[numeric_cols])

# 4. 合并数据
# 将标准化后的数值数据转换为DataFrame（保留原列名）
scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
# 合并非数值列和标准化后的数值列
final_df = pd.concat([data[non_numeric_cols], scaled_df], axis=1)

# 5. 保存处理后的数据（保留类名）
output_path = r"C:\Users\17958\Desktop\total_scaled.xlsx"
final_df.to_excel(output_path, index=False, engine='openpyxl')

print("标准化完成，文件已保存至:", output_path)