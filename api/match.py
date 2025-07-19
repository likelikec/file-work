import pandas as pd

# 项目前缀列表
prefixes = ['cli', 'csv', 'gson', 'lang', 'ruler', 'dat',"jfreechart"]

# 读取包含所有项目数据的Excel文件
df = pd.read_excel(r"C:\Users\17958\Desktop\nom-llm.xlsx")

# 创建Excel写入对象（将所有结果保存到同一个文件的不同sheet）
output_path = r"C:\Users\17958\Desktop\aligned_all.xlsx"
with pd.ExcelWriter(output_path) as writer:
    for prefix in prefixes:
        # 动态生成列名
        class_col = f"{prefix}class"
        tool_col = f"{prefix}tool"
        class_1_col = f"{prefix}class-1"
        tool_1_col = f"{prefix}tool-1"

        # 创建基准数据集
        benchmark = df[[class_1_col, tool_1_col]].copy()

        # 创建映射字典（过滤空值）
        class_tool_mapping = (
            df.dropna(subset=[class_col])
            .set_index(class_col)[tool_col]
            .to_dict()
        )

        # 执行左连接合并
        merged = pd.merge(
            left=benchmark,
            right=df[[class_col, tool_col]],
            left_on=class_1_col,
            right_on=class_col,
            how='left',
            suffixes=('_benchmark', '_original')
        )


        # 对齐处理函数（使用闭包捕获当前前缀变量）
        def align_row(row, class_col=class_col, tool_col=tool_col,
                      class_1_col=class_1_col, mapping=class_tool_mapping,
                      benchmark=benchmark):
            if not pd.isna(row[class_col]):
                # 验证是否存在于基准数据
                if row[class_col] in benchmark[class_1_col].values:
                    row[tool_col] = mapping.get(row[class_col], pd.NA)
                else:
                    row[class_col] = pd.NA
                    row[tool_col] = pd.NA
            return row


        # 应用处理逻辑
        processed = merged.apply(align_row, axis=1)

        # 清理无效数据（优化后的过滤条件）
        final_df = processed[
            processed[class_col].isin(benchmark[class_1_col]) |
            processed[class_col].isna()
            ]

        # 重组列顺序并重置索引
        final_df = final_df[
            [class_col, tool_col, class_1_col, tool_1_col]
        ].reset_index(drop=True)

        # 保存到单独的工作表
        final_df.to_excel(
            writer,
            sheet_name=f"{prefix}_aligned",
            index=False
        )

        print(f"{prefix} 项目处理完成，结果已保存至 {output_path} 的 {prefix}_aligned sheet")