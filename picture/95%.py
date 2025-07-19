import matplotlib.pyplot as plt
import pandas as pd


# testart
# data = pd.DataFrame({
#     'Variable': [
#         'COM_RAT', 'Cyclic', 'Dcy*', 'DPT*', 'LCOM', 'Level', 'PDpt', 'CLOC',
#         'INNER', 'jf', 'JLOC', 'Jm', 'Level*',  'String processing',
#         'Business Logic'
#     ],
#     'Odds Ratio': [
#         0.597714182, 0.374234405, 0.531498097, 0.765244667, 0.821525723,
#         1.220789922, 1.187742258, 0.493590169,  1.235508959,
#         0.652467296, 0.241626138, 0.792094364, 0.302491933,
#         0.769445965, 0.819102807
#     ],
#     'OR_lower': [
#         0.45, 0.27, 0.45, 0.66, 0.58, 1.08, 1.04, 0.21,  1.10,
#         0.56, 0.04, 0.65, 0.23, 0.67, 0.70
#     ],
#     'OR_upper': [
#         0.72, 0.47, 0.63, 0.89, 1.00, 1.38, 1.35, 0.74,  1.41,
#         0.76, 0.51, 0.91, 0.37,  0.89, 0.94
#     ]
# })



#sym
data = pd.DataFrame({
    'Variable': [
        'COM_RAT', 'Cyclic', 'Dcy*', 'DPT*', 'LCOM', 'Level', 'PDpt', 'CLOC',
        'INNER', 'jf', 'JLOC', 'Jm', 'Level*', 'String processing',
        'Business Logic'
    ],
    'Odds Ratio': [
        0.493167898, 0.413312058, 0.5565632, 0.756724521, 0.65135112,
        1.238453917, 1.159335845, 0.429885289, 1.269911942, 0.602144,
        0.205385028, 0.642712105, 0.327632567, 0.799907676,
        0.885927533
    ],
    'OR_lower': [
        0.35, 0.29, 0.46, 0.65, 0.46, 1.09, 1.00, 0.13, 1.12, 0.51,
        0.01, 0.49, 0.25, 0.69,  0.76
    ],
    'OR_upper': [
        0.64, 0.52, 0.65, 0.88, 0.81, 1.41, 1.32, 0.72, 1.45, 0.70,
        0.55, 0.78, 0.40, 0.93,  0.99
    ]
})




# 计算中间的 OR 值（用于绘图）
data['OR'] = (data['OR_lower'] + data['OR_upper']) / 2

# 按 OR 排序
data.sort_values('OR', inplace=True)

# 调整图片尺寸为更小的比例（原10,12 → 改为8,10）
fig, ax = plt.subplots(figsize=(8, 10))
ax.errorbar(
    data['OR'], data['Variable'],
    xerr=[data['OR'] - data['OR_lower'], data['OR_upper'] - data['OR']],
    fmt='o', capsize=6, color='darkblue', ecolor='skyblue',
    elinewidth=3, markeredgewidth=3
)

# 添加 OR=1 的参考线
ax.axvline(1, linestyle='--', color='red', linewidth=2)

# 适当缩小字体大小以适应更小的图幅
ax.set_xlabel('Odds Ratio (OR)', fontsize=14)
ax.set_ylabel('Variable', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_title('SymPrompt: OR and 95% CI', fontsize=18, pad=20)
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.set_xlim(0, 1.6)

# 美化边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 保存为PDF格式（替换原PNG格式），保持高清
plt.savefig('or_forest_plot.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()