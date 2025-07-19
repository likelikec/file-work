import re

# 输入数据
data = """
cli: BC=77.0% (94/122), LC=94.8% (146/154)
csv: BC=57.4% (89/155), LC=62.7% (106/169)
dat: BC=61.1% (196/321), LC=84.7% (643/759)
gson: BC=71.1% (241/339), LC=65.5% (294/449)
jfree: BC=63.6% (2353/3697), LC=74.9% (5871/7838)
lang: BC=73.8% (1482/2009), LC=84.4% (1997/2367)
ruler: BC=92.5% (37/40), LC=99.1% (107/108)
"""

# 权重配置
weights = {
    'cli': 4/203,
    'csv': 2/203,
    'dat': 54/203,
    'gson': 10/203,
    'jfree': 107/203,
    'lang': 19/203,
    'ruler': 7/203
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