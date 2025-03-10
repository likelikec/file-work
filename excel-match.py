import pandas as pd


def sync_excel_with_header():
    # 文件路径
    output_excel = r"C:\Users\17958\Desktop\output.xlsx"
    csv_file = r"C:\Users\17958\Desktop\指标.csv"

    # -------------------- 更新后的配置 --------------------
    TABLE1_MATCH_COLUMN = "class"
    TABLE1_OUTPUT_COLUMNS = {
        # 原有列配置
         'V': 'OCavg',
        'W': 'OCmax',
        'X': 'WMC',
        'Y':'CLOC',
        'Z': 'JLOC',
        'AA': 'LOC',
        'AB': 'JF',
        'AC': 'JM'
    }

    TABLE2_MATCH_COLUMN = "class-1"
    TABLE2_DATA_COLUMNS = {
        # 原有列配置
        'B': 'OCavg',
        'C': 'OCmax',
        'D': 'WMC',
        'E': 'CLOC',
        'F': 'JLOC',
        'G': 'LOC',
        'H': 'JF',
        'I': 'JM'
    }
    # ----------------------------------------------------

    # 读取数据（保持不变）
    df_output = pd.read_excel(output_excel, engine='openpyxl')
    df_111 = pd.read_csv(csv_file)

    # 清理列名（保持不变）
    df_output.columns = df_output.columns.str.strip()
    df_111.columns = df_111.columns.str.strip()

    # 调试输出（建议查看列名匹配情况）
    print("[调试] 表一列名:", df_output.columns.tolist())
    print("[调试] 表二列名:", df_111.columns.tolist())

    # 构建映射字典（修改为包含全部8列）
    mapping = df_111.set_index(TABLE2_MATCH_COLUMN)[
        [TABLE2_DATA_COLUMNS['B'],
         TABLE2_DATA_COLUMNS['C'],
         TABLE2_DATA_COLUMNS['D'],
         TABLE2_DATA_COLUMNS['E'],  # 新增列E
         TABLE2_DATA_COLUMNS['F'],  # 新增列F
         TABLE2_DATA_COLUMNS['G'],  # 新增列G
         TABLE2_DATA_COLUMNS['H'],  # 新增列H
         TABLE2_DATA_COLUMNS['I']  # 新增列I
         ]
    ].apply(tuple, axis=1).to_dict()

    # 数据同步（新增5列的映射）
    # 使用包含8个元素的元组，索引3-7对应新增的5列
    df_output[TABLE1_OUTPUT_COLUMNS['V']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[0]
    )
    df_output[TABLE1_OUTPUT_COLUMNS['W']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[1]
    )
    df_output[TABLE1_OUTPUT_COLUMNS['X']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[2]
    )
    # 新增列的映射
    df_output[TABLE1_OUTPUT_COLUMNS['Y']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[3]  # 索引3对应E列
    )
    df_output[TABLE1_OUTPUT_COLUMNS['Z']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[4]  # 索引4对应F列
    )
    df_output[TABLE1_OUTPUT_COLUMNS['AA']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[5]  # 索引5对应G列
    )
    df_output[TABLE1_OUTPUT_COLUMNS['AB']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[6]  # 索引6对应H列
    )
    df_output[TABLE1_OUTPUT_COLUMNS['AC']] = df_output[TABLE1_MATCH_COLUMN].map(
        lambda x: mapping.get(str(x), (None,) * 8)[7]  # 索引7对应I列
    )

    # 保存结果（保持不变）
    df_output.to_excel(
        r"C:\Users\17958\Desktop\output_modified.xlsx",
        index=False,
        engine='openpyxl'
    )
    print("同步成功！新增5列数据已写入")


if __name__ == "__main__":
    sync_excel_with_header()