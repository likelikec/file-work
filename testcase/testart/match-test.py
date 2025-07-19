import os
import re
import shutil
from pathlib import Path

# 定义筛选条件
CONDITIONS =['XYNoteAnnotation', 'WindDataItem', 'SubCategoryAxis', 'WaferMapDataset', 'NumberTickUnitSource', 'ImageEncoderFactory', 'DataPackageResources_ru', 'StrokeSample', 'NumberTickUnit', 'TextTitle', 'DefaultCategoryDataset', 'CategoryTick', 'StandardXYToolTipGenerator', 'HistogramDataset', 'StatisticalLineAndShapeRenderer', 'ChartProgressEvent', 'DefaultNumberAxisEditor', 'DateRange', 'WaferMapRenderer', 'AttrStringUtils', 'LegendItemEntity', 'DefaultHeatMapDataset', 'DefaultFlowDataset', 'IntervalMarker', 'RendererUtils', 'KeyHandler', 'ChartEditorManager', 'CategoryLineAnnotation', 'PeriodAxisLabelInfo', 'FlowKey', 'TextBlock', 'AbstractSeriesDataset', 'HeatMapUtils', 'YIntervalDataItem', 'LegendTitle', 'MatrixSeriesCollection', 'TextBox', 'GradientBarPainter', 'UnitType', 'JFreeChartEntity', 'IntervalCategoryToolTipGenerator', 'BubbleXYItemLabelGenerator', 'AreaRendererEndType', 'G2TextMeasurer', 'JDBCPieDataset', 'StandardDialRange', 'XYLineAnnotation', 'StandardDialScale', 'HMSNumberFormat', 'StandardEntityCollection', 'DefaultOHLCDataset', 'PaintList', 'KeyToGroupMap', 'PieLabelLinkStyle', 'MeanAndStandardDeviation', 'CombinedDomainXYPlot', 'XYSeries', 'AbstractDataset', 'XYTitleAnnotation', 'PeriodAxis', 'CustomXYToolTipGenerator', 'TickUnits', 'XYCoordinateType', 'SlidingCategoryDataset', 'LegendGraphic', 'KeyedValueComparator', 'PaintAlpha', 'TimeSeriesDataItem', 'LineAndShapeRenderer', 'DatasetGroup', 'CategoryStepRenderer', 'StandardXYSeriesLabelGenerator', 'KeyedObjects2D', 'LogarithmicAxis', 'PaintMap', 'SymbolicXYItemLabelGenerator', 'RendererChangeEvent', 'DefaultIntervalCategoryDataset', 'ItemLabelPosition', 'CenterTextMode', 'Regression', 'DateTickUnit', 'AnnotationChangeEvent', 'RelativeDateFormat', 'TaskSeriesCollection', 'DefaultXYZDataset', 'DefaultKeyedValues', 'RegularTimePeriod', 'XYSeriesCollection', 'StackedXYBarRenderer', 'CustomCategoryURLGenerator', 'VectorRenderer', 'SymbolAxis', 'StandardXYBarPainter', 'StandardTickUnitSource', 'ItemLabelAnchor', 'HistogramBin', 'XYStepRenderer', 'ChartFactory', 'CategoryMarker', 'PieSectionEntity', 'LengthConstraintType', 'DatasetChangeEvent', 'XYDotRenderer', 'StandardCategorySeriesLabelGenerator', 'StackedXYAreaRenderer', 'StandardXYItemRenderer', 'DirectionalGradientPaintTransformer']


# dat
# ['Drone', 'Science', 'Horse', 'Australia', 'Supernatural', 'SqlTransformer', 'RockBand', 'Demographic', 'Esports', 'FoodFaker', 'HeyArnold', 'Tea', 'Photography', 'Medical', 'Hashing', 'DateAndTime', 'RuPaulDragRace', 'Schema', 'SuperSmashBros', 'ClashOfClans', 'DungeonsAndDragons', 'OlympicSport', 'ElderScrolls', 'CompositeField', 'University', 'SoulKnight', 'Chiquito', 'Cricket', 'BackToTheFuture', 'TheExpanse', 'DarkSouls', 'Control', 'FullmetalAlchemist', 'SingletonLocale', 'Zelda', 'Team', 'ProgrammingLanguage', 'FakeResolver', 'FreshPrinceOfBelAir', 'WordUtils', 'Artist', 'Military', 'TheRoom', 'Hobbit', 'Babylon5', 'Volleyball', 'SwordArtOnline', 'HowToTrainYourDragon', 'OscarMovie', 'FamousLastWords', 'Formula1', 'Disease', 'BreakingBad', 'FakeValuesGrouping', 'CultureSeries']



    # lang
    # ['LocaleUtils', 'CharSequenceTranslator', 'FastDateParser', 'CompareToBuilder', 'MutableInt', 'FastDateFormat', 'ConcurrentRuntimeException', 'MutableByte', 'Conversion', 'MutableLong', 'DateUtils', 'StandardToStringStyle', 'BasicThreadFactory', 'ToStringStyle', 'SerializationException', 'ConcurrentException', 'EntityArrays', 'DateFormatUtils', 'EqualsBuilder', 'IEEE754rUtils']

    # jfree
    # ['XYNoteAnnotation', 'WindDataItem', 'SubCategoryAxis', 'WaferMapDataset', 'NumberTickUnitSource', 'ImageEncoderFactory', 'DataPackageResources_ru', 'StrokeSample', 'NumberTickUnit', 'TextTitle', 'DefaultCategoryDataset', 'CategoryTick', 'StandardXYToolTipGenerator', 'HistogramDataset', 'StatisticalLineAndShapeRenderer', 'ChartProgressEvent', 'DefaultNumberAxisEditor', 'DateRange', 'WaferMapRenderer', 'AttrStringUtils', 'LegendItemEntity', 'DefaultHeatMapDataset', 'DefaultFlowDataset', 'IntervalMarker', 'RendererUtils', 'KeyHandler', 'ChartEditorManager', 'CategoryLineAnnotation', 'PeriodAxisLabelInfo', 'FlowKey', 'TextBlock', 'AbstractSeriesDataset', 'HeatMapUtils', 'YIntervalDataItem', 'LegendTitle', 'MatrixSeriesCollection', 'TextBox', 'GradientBarPainter', 'UnitType', 'JFreeChartEntity', 'IntervalCategoryToolTipGenerator', 'BubbleXYItemLabelGenerator', 'AreaRendererEndType', 'G2TextMeasurer', 'JDBCPieDataset', 'StandardDialRange', 'XYLineAnnotation', 'StandardDialScale', 'HMSNumberFormat', 'StandardEntityCollection', 'DefaultOHLCDataset', 'PaintList', 'KeyToGroupMap', 'PieLabelLinkStyle', 'MeanAndStandardDeviation', 'CombinedDomainXYPlot', 'XYSeries', 'AbstractDataset', 'XYTitleAnnotation', 'PeriodAxis', 'CustomXYToolTipGenerator', 'TickUnits', 'XYCoordinateType', 'SlidingCategoryDataset', 'LegendGraphic', 'KeyedValueComparator', 'PaintAlpha', 'TimeSeriesDataItem', 'LineAndShapeRenderer', 'DatasetGroup', 'CategoryStepRenderer', 'StandardXYSeriesLabelGenerator', 'KeyedObjects2D', 'LogarithmicAxis', 'PaintMap', 'SymbolicXYItemLabelGenerator', 'RendererChangeEvent', 'DefaultIntervalCategoryDataset', 'ItemLabelPosition', 'CenterTextMode', 'Regression', 'DateTickUnit', 'AnnotationChangeEvent', 'RelativeDateFormat', 'TaskSeriesCollection', 'DefaultXYZDataset', 'DefaultKeyedValues', 'RegularTimePeriod', 'XYSeriesCollection', 'StackedXYBarRenderer', 'CustomCategoryURLGenerator', 'VectorRenderer', 'SymbolAxis', 'StandardXYBarPainter', 'StandardTickUnitSource', 'ItemLabelAnchor', 'HistogramBin', 'XYStepRenderer', 'ChartFactory', 'CategoryMarker', 'PieSectionEntity', 'LengthConstraintType', 'DatasetChangeEvent', 'XYDotRenderer', 'StandardCategorySeriesLabelGenerator', 'StackedXYAreaRenderer', 'StandardXYItemRenderer', 'DirectionalGradientPaintTransformer']

# [
    # "BitField", "BooleanUtils", "EqualsBuilder", "IDKey",
    # "StandardToStringStyle", "ConcurrentException",
    # "ContextedRuntimeException", "MethodUtils", "TypeUtils",
    # "SerializationUtils", "StrBuilder", "AggregateTranslator",
    # "CharSequenceTranslator", "JavaUnicodeEscaper", "DateUtils",
    # "FastDateFormat", "FastDateParser", "FormatCache", "ImmutablePair"
#
# "CollectionTypeAdapterFactory",
# "NumberTypeAdapter",
# "TreeTypeAdapter",
# "Excluder",
# "NonNullElementWrapperList",
# "PreJava9DateFormatProvider",
# "Primitives",
# "ReflectionHelper",
# "Streams",
# "JsonObject",
# "JsonPrimitive"

# ]
#

def extract_class_name(filename):
    """从文件名中提取类名，改进正则表达式匹配"""
    # 匹配以大写字母开头的类名，直到遇到非字母数字字符或下划线
    match = re.match(r'^([A-Z][a-zA-Z0-9]*)(?:_|\.|$)', filename)
    return match.group(1) if match else None


def should_include_file(filename):
    """判断文件是否应被包含"""
    class_name = extract_class_name(filename)
    print(f"检查文件: {filename} -> 提取的类名: {class_name}")
    return class_name in CONDITIONS


def copy_matching_files(source_dir, target_dir, log_file):
    """复制匹配的文件及其包结构"""
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 记录匹配的类及其文件数量
    matched_classes = {}
    # 记录未匹配的类
    unmatched_classes = set(CONDITIONS.copy())

    # 遍历源目录中的所有文件和文件夹
    print(f"扫描源目录: {source_dir}")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"源目录: {source_dir}\n")
        f.write("匹配结果:\n")

        for root, dirs, files in os.walk(source_dir):
            print(f"  检查目录: {root}")
            f.write(f"\n检查目录: {root}\n")
            # 确定相对路径
            relative_path = os.path.relpath(root, source_dir)
            target_subdir = os.path.join(target_dir, relative_path)

            # 创建对应的目标子目录
            os.makedirs(target_subdir, exist_ok=True)

            # 复制匹配的文件
            for file in files:
                if file.endswith('.java'):  # 只处理Java文件
                    class_name = extract_class_name(file)
                    if class_name in CONDITIONS:
                        source_file = os.path.join(root, file)
                        target_file = os.path.join(target_subdir, file)
                        shutil.copy2(source_file, target_file)

                        # 更新匹配的类的计数
                        if class_name in matched_classes:
                            matched_classes[class_name] += 1
                        else:
                            matched_classes[class_name] = 1

                        # 从未匹配集合中移除
                        if class_name in unmatched_classes:
                            unmatched_classes.remove(class_name)

                        log_msg = f"    复制: {source_file} -> {target_file}"
                        print(log_msg)
                        f.write(log_msg + "\n")
                    else:
                        log_msg = f"    跳过: {file} (类名: {class_name} 不匹配条件)"
                        print(log_msg)
                        f.write(log_msg + "\n")
                else:
                    log_msg = f"    跳过非Java文件: {file}"
                    print(log_msg)
                    f.write(log_msg + "\n")

        # 写入匹配统计
        f.write("\n匹配统计:\n")
        for class_name, count in matched_classes.items():
            f.write(f"  - {class_name}: {count}个文件\n")

        # 写入未匹配的类
        f.write("\n未匹配的类:\n")
        if unmatched_classes:
            for class_name in unmatched_classes:
                f.write(f"  - {class_name}\n")
        else:
            f.write("  无\n")

    return matched_classes, unmatched_classes


def main():
    # 定义源目录和目标目录
    source_dir1 = r"C:\Users\17958\Desktop\TestCases-defect4j\evo\jfree"
    source_dir2 = r"C:\Users\17958\Desktop\TestCases-defect4j\testart\jfree"

    target_dir1 = r"C:\Users\17958\Desktop\jfree项目\evo-testart"
    target_dir2 = r"C:\Users\17958\Desktop\jfree项目\testart"

    # 定义日志文件路径
    log_file1 = r"C:\Users\17958\Desktop\jfree项目\evo-testart-matches.txt"
    log_file2 = r"C:\Users\17958\Desktop\jfree项目\testart-matches.txt"

    # 检查源目录是否存在
    if not os.path.exists(source_dir1):
        print(f"错误: 源目录1不存在 - {source_dir1}")
        return

    if not os.path.exists(source_dir2):
        print(f"错误: 源目录2不存在 - {source_dir2}")
        return

    # 复制文件并获取匹配结果
    print("正在复制文件夹1的内容...")
    folder1_matches, folder1_unmatched = copy_matching_files(source_dir1, target_dir1, log_file1)
    print(f"文件夹1匹配了 {len(folder1_matches)} 个类，共 {sum(folder1_matches.values())} 个文件")
    for class_name, count in folder1_matches.items():
        print(f"  - {class_name}: {count}个文件")

    print("\n文件夹1未匹配的类:")
    if folder1_unmatched:
        for class_name in folder1_unmatched:
            print(f"  - {class_name}")
    else:
        print("  无")

    print("\n正在复制文件夹2的内容...")
    folder2_matches, folder2_unmatched = copy_matching_files(source_dir2, target_dir2, log_file2)
    print(f"文件夹2匹配了 {len(folder2_matches)} 个类，共 {sum(folder2_matches.values())} 个文件")
    for class_name, count in folder2_matches.items():
        print(f"  - {class_name}: {count}个文件")

    print("\n文件夹2未匹配的类:")
    if folder2_unmatched:
        for class_name in folder2_unmatched:
            print(f"  - {class_name}")
    else:
        print("  无")

    print(f"\n匹配结果已保存到:\n- {log_file1}\n- {log_file2}")
    print("操作完成!")


if __name__ == "__main__":
    main()