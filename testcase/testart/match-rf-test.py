import os
import re
import shutil
from pathlib import Path

# 定义筛选条件及来源映射（类名: (来源文件夹编号, 计数)）
CONDITIONS ={
    "Patterns": (1, 0),
    "CompositeByteTransition": (1, 0),
    "ByteState": (1, 0),
    "Range": (1, 0),
    "InputByte": (2, 1),
    "InputWildcard": (2, 1),
    "ParseException": (2, 1),
    "ByteMatch": (1, 0),
    "IntIntMap": (1, 0),
    "XYNoteAnnotation": (2, 1),
    "WindDataItem": (2, 1),
    "SubCategoryAxis": (2, 1),
    "WaferMapDataset": (2, 1),
    "NumberTickUnitSource": (2, 1),
    "ImageEncoderFactory": (2, 1),
    "DataPackageResources_ru": (2, 1),
    "StrokeSample": (2, 1),
    "NumberTickUnit": (2, 1),
    "TextTitle": (2, 1),
    "DefaultCategoryDataset": (2, 1),
    "CategoryTick": (2, 1),
    "StandardXYToolTipGenerator": (2, 1),
    "HistogramDataset": (2, 1),
    "StatisticalLineAndShapeRenderer": (1, 0),
    "ChartProgressEvent": (2, 1),
    "DefaultNumberAxisEditor": (2, 1),
    "DateRange": (2, 1),
    "WaferMapRenderer": (2, 1),
    "AttrStringUtils": (2, 1),
    "LegendItemEntity": (2, 1),
    "DefaultHeatMapDataset": (2, 1),
    "DefaultFlowDataset": (2, 1),
    "IntervalMarker": (2, 1),
    "RendererUtils": (2, 1),
    "KeyHandler": (2, 1),
    "ChartEditorManager": (2, 1),
    "CategoryLineAnnotation": (2, 1),
    "PeriodAxisLabelInfo": (2, 1),
    "FlowKey": (2, 1),
    "TextBlock": (2, 1),
    "AbstractSeriesDataset": (1, 0),
    "HeatMapUtils": (1, 0),
    "YIntervalDataItem": (2, 1),
    "LegendTitle": (2, 1),
    "MatrixSeriesCollection": (2, 1),
    "TextBox": (2, 1),
    "GradientBarPainter": (2, 1),
    "UnitType": (1, 0),
    "JFreeChartEntity": (2, 1),
    "IntervalCategoryToolTipGenerator": (2, 1),
    "BubbleXYItemLabelGenerator": (2, 1),
    "AreaRendererEndType": (2, 1),
    "G2TextMeasurer": (2, 1),
    "JDBCPieDataset": (2, 1),
    "StandardDialRange": (2, 1),
    "XYLineAnnotation": (1, 0),
    "StandardDialScale": (2, 1),
    "HMSNumberFormat": (2, 1),
    "StandardEntityCollection": (2, 1),
    "DefaultOHLCDataset": (2, 1),
    "PaintList": (2, 1),
    "KeyToGroupMap": (2, 1),
    "PieLabelLinkStyle": (2, 1),
    "MeanAndStandardDeviation": (2, 1),
    "CombinedDomainXYPlot": (2, 1),
    "XYSeries": (2, 1),
    "AbstractDataset": (1, 0),
    "XYTitleAnnotation": (2, 1),
    "PeriodAxis": (2, 1),
    "CustomXYToolTipGenerator": (2, 1),
    "TickUnits": (2, 1),
    "XYCoordinateType": (2, 1),
    "SlidingCategoryDataset": (2, 1),
    "LegendGraphic": (2, 1),
    "KeyedValueComparator": (2, 1),
    "PaintAlpha": (2, 1),
    "TimeSeriesDataItem": (2, 1),
    "LineAndShapeRenderer": (2, 1),
    "DatasetGroup": (2, 1),
    "CategoryStepRenderer": (1, 0),
    "StandardXYSeriesLabelGenerator": (2, 1),
    "KeyedObjects2D": (2, 1),
    "LogarithmicAxis": (1, 0),
    "PaintMap": (2, 1),
    "SymbolicXYItemLabelGenerator": (2, 1),
    "RendererChangeEvent": (2, 1),
    "DefaultIntervalCategoryDataset": (2, 1),
    "ItemLabelPosition": (2, 1),
    "CenterTextMode": (1, 0),
    "Regression": (1, 0),
    "DateTickUnit": (2, 1),
    "AnnotationChangeEvent": (2, 1),
    "RelativeDateFormat": (2, 1),
    "TaskSeriesCollection": (2, 1),
    "DefaultXYZDataset": (2, 1),
    "DefaultKeyedValues": (2, 1),
    "RegularTimePeriod": (1, 0),
    "XYSeriesCollection": (2, 1),
    "StackedXYBarRenderer": (2, 1),
    "CustomCategoryURLGenerator": (2, 1),
    "VectorRenderer": (2, 1),
    "SymbolAxis": (2, 1),
    "StandardXYBarPainter": (2, 1),
    "StandardTickUnitSource": (2, 1),
    "ItemLabelAnchor": (2, 1),
    "HistogramBin": (2, 1),
    "XYStepRenderer": (2, 1),
    "ChartFactory": (1, 0),
    "CategoryMarker": (2, 1),
    "PieSectionEntity": (2, 1),
    "LengthConstraintType": (2, 1),
    "DatasetChangeEvent": (2, 1),
    "XYDotRenderer": (2, 1),
    "StandardCategorySeriesLabelGenerator": (2, 1),
    "StackedXYAreaRenderer": (1, 0),
    "StandardXYItemRenderer": (2, 1),
    "DirectionalGradientPaintTransformer": (2, 1),
}

def extract_class_name(filename):
    """从文件名中提取类名，改进正则表达式匹配"""
    # 匹配以大写字母开头的类名，直到遇到非字母数字字符或下划线
    match = re.match(r'^([A-Z][a-zA-Z0-9]*)(?:_|\.|$)', filename)
    return match.group(1) if match else None


def copy_matching_files(source_dir, target_dir, class_whitelist=None):
    """复制匹配的文件及其包结构
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        class_whitelist: 只复制这些类名的文件，如果为None则复制所有
    Returns:
        匹配的类及其文件数量的字典
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 记录匹配的类及其文件数量
    matched_classes = {}

    # 遍历源目录中的所有文件和文件夹
    print(f"扫描源目录: {source_dir}")
    for root, dirs, files in os.walk(source_dir):
        print(f"  检查目录: {root}")
        # 确定相对路径
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)

        # 创建对应的目标子目录
        os.makedirs(target_subdir, exist_ok=True)

        # 复制匹配的文件
        for file in files:
            if file.endswith('.java'):  # 只处理Java文件
                class_name = extract_class_name(file)
                if class_whitelist is None or class_name in class_whitelist:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_subdir, file)
                    shutil.copy2(source_file, target_file)

                    # 更新匹配的类的计数
                    if class_name in matched_classes:
                        matched_classes[class_name] += 1
                    else:
                        matched_classes[class_name] = 1
                    print(f"    复制: {source_file} -> {target_file}")
                else:
                    print(f"    跳过: {file} (类名: {class_name} 不在白名单中)")
            else:
                print(f"    跳过非Java文件: {file}")

    return matched_classes


def write_matched_classes(all_matches, output_file):
    """将匹配的类名写入文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("匹配的类名统计:\n")
        f.write("=" * 30 + "\n")
        for class_name, count in sorted(all_matches.items()):
            f.write(f"{class_name}: {count}个文件\n")


def main():
    # 定义源目录和目标目录
    source_dir1 = r"C:\Users\17958\Desktop\TestCases-defect4j\evo\jfree"
    source_dir2 = r"C:\Users\17958\Desktop\TestCases-defect4j\testart\jfree"

    target_dir_sym_rf = r"C:\Users\17958\Desktop\jfree项目\testart-rf"
    output_file = r"C:\Users\17958\Desktop\jfree项目\testart-rf.txt"

    # 按来源分组类
    source1_classes = [cls for cls, (src, cnt) in CONDITIONS.items() if src == 1 and cnt == 0]
    source2_classes = [cls for cls, (src, cnt) in CONDITIONS.items() if src == 2 and cnt == 1]

    print(f"从文件夹1提取的类: {source1_classes}")
    print(f"从文件夹2提取的类: {source2_classes}")

    # 检查源目录是否存在
    if not os.path.exists(source_dir1):
        print(f"错误: 源目录1不存在 - {source_dir1}")
        return

    if not os.path.exists(source_dir2):
        print(f"错误: 源目录2不存在 - {source_dir2}")
        return

    # 从文件夹1复制计数为0的类
    print("\n正在从文件夹1复制计数为0的类...")
    folder1_matches = copy_matching_files(source_dir1, target_dir_sym_rf, source1_classes)

    # 从文件夹2复制计数为1的类
    print("\n正在从文件夹2复制计数为1的类...")
    folder2_matches = copy_matching_files(source_dir2, target_dir_sym_rf, source2_classes)

    # 合并匹配结果
    all_matches = folder1_matches.copy()
    for cls, cnt in folder2_matches.items():
        if cls in all_matches:
            all_matches[cls] += cnt
        else:
            all_matches[cls] = cnt

    # 找出没有匹配上的类
    unmatched_classes = [cls for cls in CONDITIONS if cls not in all_matches or all_matches[cls] == 0]

    # 输出统计信息
    print("\n匹配统计:")
    print(f"共匹配了 {len(all_matches)} 个类，共 {sum(all_matches.values())} 个文件")
    for class_name in CONDITIONS:
        count = all_matches.get(class_name, 0)
        source = CONDITIONS[class_name][0]
        status = "匹配成功" if count > 0 else "未匹配到任何文件"
        print(f"  - {class_name}: 从文件夹{source}复制了 {count} 个文件 ({status})")

    # 输出未匹配的类
    if unmatched_classes:
        print("\n未匹配到任何文件的类:")
        for cls in unmatched_classes:
            print(f"  - {cls}")
    else:
        print("\n所有类都成功匹配到文件!")

    # 写入匹配结果到文件
    write_matched_classes(all_matches, output_file)
    print(f"\n匹配结果已保存到: {output_file}")

    print("\n操作完成!")


if __name__ == "__main__":
    main()