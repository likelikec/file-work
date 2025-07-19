def calculate_weighted_average():
    # 原始格式数据直接嵌入代码
    data = '''
   cli: BC=77.0% (94/122), LC=94.8% (146/154)
csv: BC=57.4% (89/155), LC=62.7% (106/169)
dat: BC=61.1% (196/321), LC=84.7% (643/759)
gson: BC=71.1% (241/339), LC=65.5% (294/449)
jfree: BC=63.6% (2353/3697), LC=74.9% (5871/7838)
lang: BC=73.8% (1482/2009), LC=84.4% (1997/2367)
ruler: BC=92.5% (37/40), LC=99.1% (107/108)
    '''
    # 初始化存储列表
    bc_numerators = []
    bc_denominators = []
    lc_numerators = []
    lc_denominators = []

    # 正确处理数据：按行分割
    for line in data.strip().split('\n'):
        line = line.strip()
        if not line:  # 跳过空行
            continue

        try:
            # 提取项目名称
            project_name = line.split(':')[0].strip()

            # 提取BC和LC部分
            # 更灵活的分割方式，支持多种格式
            bc_start = line.index("BC=")
            lc_start = line.index("LC=")

            # 提取BC部分
            bc_part_raw = line[bc_start:lc_start].strip()
            bc_value_part = bc_part_raw.split('=')[1].strip()
            bc_fraction = bc_value_part.split('(')[1].split(')')[0].strip()
            bc_num, bc_den = map(int, bc_fraction.split('/'))

            # 提取LC部分
            lc_part_raw = line[lc_start:].strip()
            lc_value_part = lc_part_raw.split('=')[1].strip()
            lc_fraction = lc_value_part.split('(')[1].split(')')[0].strip()
            lc_num, lc_den = map(int, lc_fraction.split('/'))

            # 存储数据
            bc_numerators.append(bc_num)
            bc_denominators.append(bc_den)
            lc_numerators.append(lc_num)
            lc_denominators.append(lc_den)

            print(f"✅ 已处理项目: {project_name} - BC: {bc_num}/{bc_den}, LC: {lc_num}/{lc_den}")

        except Exception as e:
            print(f"❌ 解析错误: 无法处理行 '{line}' - 错误: {str(e)}")
            continue  # 继续处理下一行

    # 检查是否有有效数据
    if not bc_numerators or not lc_numerators:
        print("错误: 没有找到有效的数据进行计算")
        return

    # 计算加权平均值
    total_bc_num = sum(bc_numerators)
    total_bc_den = sum(bc_denominators)
    bc_weighted_avg = (total_bc_num / total_bc_den) * 100 if total_bc_den != 0 else 0

    total_lc_num = sum(lc_numerators)
    total_lc_den = sum(lc_denominators)
    lc_weighted_avg = (total_lc_num / total_lc_den) * 100 if total_lc_den != 0 else 0

    # 输出结果
    print("\n计算结果：")
    print(f"BC的加权平均值：{bc_weighted_avg:.2f}%（{total_bc_num}/{total_bc_den}）")
    print(f"LC的加权平均值：{lc_weighted_avg:.2f}%（{total_lc_num}/{total_lc_den}）")


if __name__ == "__main__":
    calculate_weighted_average()