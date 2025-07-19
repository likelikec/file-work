import os
import shutil
from pathlib import Path


def main():
    # 源文件夹路径
    source_dir = r"E:\unit-generate\google-json - evo\src\main\java\com"
    # 目标文件夹路径
    target_dir = os.path.join(os.path.dirname(source_dir), "select")

    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    # 要筛选的类名列表
    class_names = ['JsonObject', 'JsonPrimitive', 'Primitives', 'PreJava9DateFormatProvider', 'Streams', 'NonNullElementWrapperList', 'NumberTypeAdapter', 'CollectionTypeAdapterFactory', 'Excluder', 'ReflectionHelper', 'TreeTypeAdapter']


    # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".java"):
                # 获取类名（文件名去掉.java后缀）
                class_name = os.path.splitext(file)[0]

                # 检查类名是否在目标列表中
                if class_name in class_names:
                    # 构建源文件的完整路径
                    source_file = os.path.join(root, file)

                    # 计算目标文件的相对路径
                    relative_path = os.path.relpath(root, source_dir)
                    target_subdir = os.path.join(target_dir, relative_path)

                    # 创建目标子文件夹
                    os.makedirs(target_subdir, exist_ok=True)

                    # 复制文件
                    target_file = os.path.join(target_subdir, file)
                    shutil.copy2(source_file, target_file)
                    print(f"已复制: {source_file} -> {target_file}")

    print("筛选完成！")


if __name__ == "__main__":
    main()