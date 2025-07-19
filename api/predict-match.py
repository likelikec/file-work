import pandas as pd


def match_and_filter_excel(file_a_path, file_b_path, output_path):
    try:
        # 读取文件a和文件b
        df_a = pd.read_excel(file_a_path)
        df_b = pd.read_excel(file_b_path)

        # 检查文件a和文件b是否包含必要的列
        required_columns_a = ['class', 'project', 'yes']
        required_columns_b = ['Class Name', 'project']

        for col in required_columns_a:
            if col not in df_a.columns:
                raise ValueError(f"文件a中缺少必要的列: {col}")

        for col in required_columns_b:
            if col not in df_b.columns:
                raise ValueError(f"文件b中缺少必要的列: {col}")

        # 创建一个基于文件a的(class, project)到yes值的映射
        a_key_to_yes = {(row['class'], row['project']): row['yes']
                        for _, row in df_a.iterrows()}

        # 筛选文件b并添加yes列
        filtered_rows = []
        for _, row in df_b.iterrows():
            key = (row['Class Name'], row['project'])
            if key in a_key_to_yes:
                new_row = row.copy()
                new_row['yes'] = a_key_to_yes[key]
                filtered_rows.append(new_row)

        # 创建筛选后的DataFrame并调整列顺序
        df_b_filtered = pd.DataFrame(filtered_rows)

        # 确保project列后是yes列
        if not df_b_filtered.empty:
            cols = df_b_filtered.columns.tolist()
            project_idx = cols.index('project')
            cols.insert(project_idx + 1, 'yes')
            df_b_filtered = df_b_filtered[cols]

        # 保存筛选后的结果
        df_b_filtered.to_excel(output_path, index=False)
        print(f"筛选完成，结果已保存至: {output_path}")

        return df_b_filtered

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    # 文件路径配置
    file_a_path = r"C:\Users\17958\Desktop\sym-pre.xlsx"
    file_b_path = r"C:\Users\17958\Desktop\gemini-14-classification_results_合并.xlsx"
    output_path = r"C:\Users\17958\Desktop\all-res.xlsx"

    # 执行匹配和筛选
    match_and_filter_excel(file_a_path, file_b_path, output_path)