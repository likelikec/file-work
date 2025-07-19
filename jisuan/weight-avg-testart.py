import re

# 输入数据
data = """
cli: BC=79.4% (54/68), LC=91.7% (132/144)
csv: BC=68.0% (34/50), LC=91.4% (64/70)
dat: BC=62.2% (201/323), LC=83.9% (635/757)
gson: BC=79.2% (262/331), LC=72.4% (322/445)
jfree: BC=65.9% (2766/4196), LC=78.9% (6759/8569)
lang: BC=70.2% (1348/1919), LC=85.3% (2076/2434)
ruler: BC=92.5% (37/40), LC=99.1% (107/108)

"""

# 权重配置
weights = {
    'cli': 4/203,
    'csv': 2/203,
    'dat': 54/203,
    'gson': 10/203,
    'jfree': 107/203,
    'lang': 20/203,
    'ruler': 6/203
}

# 解析数据
parsed_data = {}
for line in data.split('\n'):
    if not line.strip():
        continue
    # 提取项目名、BC值和LC值
    match = re.match(r'(\w+):\s+BC=(\d+\.\d+)%\s+\([^)]+\),\s+LC=(\d+\.\d+)%\s+\([^)]+\)', line)
    if match:
        project, bc, lc = match.groups()
        parsed_data[project] = {
            'bc': float(bc),
            'lc': float(lc)
        }

# 计算加权值
weighted_bc = 0
weighted_lc = 0

for project, metrics in parsed_data.items():
    weight = weights.get(project, 0)
    weighted_bc += metrics['bc'] * weight
    weighted_lc += metrics['lc'] * weight

# 输出结果
print(f"加权 BC: {weighted_bc:.2f}%  加权 LC: {weighted_lc:.2f}%")