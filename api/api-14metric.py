import os
import requests
import json
import pandas as pd

# 配置 API 密钥和 URL
API_KEY = "sk-BNBbQFIP7c4E04c33f86T3BlbKFJ4Ae550543E8b4e42b4b3"
API_URL = "https://c-z0-api-01.hash070.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
}

# 7个项目的配置（名称从路径中提取，路径已更新为您提供的地址）
PROJECT_CONFIGS = [
    {
        "name": "commons-csv",
        "path": "E:\\unit-generate\\commons-csv\\src\\main\\java\\org\\apache\\commons\\csv"
    },
    {
        "name": "commons-lang",
        "path": "E:\\unit-generate\\commons-lang\\src\\main\\java\\org\\apache\\commons\\lang3"
    },
    {
        "name": "google-json",
        "path": "E:\\unit-generate\\google-json\\src\\main\\java\\com\\google\\gson"
    },
    {
        "name": "commons-cli-evo",
        "path": "E:\\unit-generate\\commons-cli-evo\\src\\main\\java\\org\\apache\\commons\\cli"
    },
    {
        "name": "ruler",
        "path": "E:\\unit-generate\\ruler\\src\\main\\software\\amazon\\event\\ruler"
    },
    {
        "name": "restful-demo-1-dat",
        "path": "D:\\restful-demo-1\\dat\\src\\main\\java\\net\\datafaker"
    },
    {
        "name": "jfreechart154",
        "path": "E:\\unit-generate\\jfreechart154\\src\\main\\java\\org\\jfree"
    }
]


# 获取项目目录下所有 .java 文件
def get_java_files(project_dir):
    java_files = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files


# 从文件路径提取类名
def get_class_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


# 读取文件内容
def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 构造提示（保持不变）
def construct_prompt(class_name, class_code):
    prompt = f"""

You are a senior professor of software engineering. Please classify each class based on your professional knowledge and the following 25 indicators that describe code characteristics. Read the code, combine the characteristics of test case generation by Evo and LLM, calculate the 25 indicators that describe code characteristics to determine whether it is more appropriate to use LLM or Evosuite to generate test cases.

Evosuite is a tool that automatically generates Java-like test cases using evolutionary algorithms, with the goal of achieving high code coverage. LLM can generate test cases based on its understanding of the code and behavior.

The 15 metrics are as follows:

1. **Number of Transitive Dependencies (Dcy*)** : Derived from Dcy, it refers to the number of other classes that a class directly or indirectly depends on, including the number of directly dependent classes and indirectly dependent classes, which can reflect the deep coupling state of the class in the dependency network.
2. **Number of Transitive Dependents (DPT*)** : It is an object-oriented coupling feature, referring to the number of classes that indirectly depend on the current class through the dependency chain. It is obtained by constructing a dependency graph and using a graph traversal algorithm to find all transitive dependencies and then statistically analyzing them.
3. **Cyclic Dependencies ** : It is an object-oriented coupling feature, referring to the number of other classes that a class directly or indirectly depends on, and these dependencies also directly or indirectly depend on that class. It is obtained by constructing a dependency graph, collecting direct dependencies, calculating transitive dependencies, identifying strongly connected components, and then subtracting 1 after statistics, representing the number of other classes involved in circular dependencies.
4. **Level (Level)** : It is an object-oriented theoretical complexity feature that measures the "number of levels" of the classes a class depends on. When it does not depend on other classes, its value is 0; when it does depend, its value is the maximum Level value among the dependent classes plus 1 (excluding classes that are mutually dependent or cyclically dependent).
5. Adjusted Level (Level*) : It is a theoretical complexity feature of object-oriented programming. Based on the "number of layers" of the classes that a class depends on, it considers the number of classes in circular dependencies. When it does not depend on other classes, the value is 0. When there is a dependency, the value is the maximum Level* value in the dependent class (non-circular dependency) plus the number of classes that are interdependent with or form a circular dependency with that class.
6.**Package Dependents (PDpt)** : It is the dependency feature corresponding to PDcy, referring to the number of packages that directly or indirectly depend on the current class. It is obtained by counting the number of packages that directly or indirectly depend on the class.
7.Lack of Cohesion of Methods (LCOM) : It is an object-oriented cohesion feature, referring to the degree of lack of cohesion among the methods of a class. It is obtained by constructing an inter-method relationship graph (where nodes represent methods and edges represent inter-method relationships) and calculating the number of connected components in the graph, and is related to the cohesion and responsibility of the class.
8.**Comment Lines of Code (CLOC)** : It is an extension of the concept of source code lines of code (SLOC) proposed by Lazic. It refers to the total number of lines containing comment content in the code file, calculated by strict syntax parsing, reflecting the physical density of comments in the code and its relationship with the interpretability of the code.
9. **Javadoc Lines of Code (JLOC)** : It is a refined metric of CLOC, specifically counting the number of lines of comments that comply with the Javadoc specification, measuring the density of code comments that follow the conventions of the standard API documentation, and is closely related to the completeness of the API documentation.
10. **Javadoc Field Coverage (JF)** : It is a coverage feature based on JLOC, referring to the percentage of the number of fields with Javadoc comments to the total number of fields in the class, quantifying the documentation level of fields in the class.
11. **Javadoc Method Coverage (JM)** : Derived from JLOC, it refers to the percentage of the number of methods with Javadoc annotations to the total number of methods in the class, and is closely related to the integrity of the method-level documentation.
12. **Comment Ratio (COM_RAT)** : It is a code comment feature, referring to the ratio of the number of comment lines in the code to the total number of code lines (excluding blank lines), reflecting the relative density of comments in the code. It is a classic feature for evaluating code readability.
13.**Number of Implemented Interfaces (INNER)** : Defined considering the need to describe the size of a class from the perspective of interface implementation, it refers to the total number of interfaces implemented by a class, including all different interfaces implemented directly and inherited from the parent class.
14.String Processing** : This scenario involves the creation, manipulation, parsing, formatting or transformation of strings, such as processing text data, generating reports, cleaning input, as well as methods like string splitting, concatenation, and regular expression matching.
15.*Business Logic** : This scenario mainly implements specific business rules or domain logic, which is specific to the application context, such as user permission verification, workflow engine, etc.


Here is a Java class named {class_name}:

{class_code}

Please respond in the following JSON format:

{{"class_name": "{class_name}", "tool": "LLM" or "Evosuite"}}
"""
    return prompt


# 调用 API
def call_api(prompt, model="gpt-3.5-turbo"):
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "stream": False
    }

    try:
        print(f"[DEBUG] 正在调用 API 处理类...")
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=params,
            timeout=30
        )
        response.raise_for_status()
        print("[DEBUG] API 响应成功")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 调用失败: {e}")
        return None


# 处理单个项目
def process_project(project_config, model="gpt-3.5-turbo"):
    project_name = project_config["name"]
    project_dir = project_config["path"]

    print(f"\n===== 开始处理项目: {project_name} =====")
    print(f"项目路径: {project_dir}")

    # 检查项目目录是否存在
    if not os.path.exists(project_dir):
        print(f"错误：项目目录不存在 - {project_dir}")
        return [], project_name

    java_files = get_java_files(project_dir)
    print(f"找到 {len(java_files)} 个 .java 文件")

    results = []
    for file_path in java_files:
        class_name = get_class_name(file_path)
        print(f"\n处理类: {class_name}")

        try:
            class_code = read_file_content(file_path)
        except UnicodeDecodeError:
            print(f"警告：{file_path} 编码不是 utf-8，跳过该文件")
            continue
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}，跳过该文件")
            continue

        prompt = construct_prompt(class_name, class_code)
        response = call_api(prompt, model)

        if response:
            try:
                res_content = response["choices"][0]["message"]["content"]
                tool_dict = json.loads(res_content)
                results.append([tool_dict["class_name"], tool_dict["tool"]])
                print(f"已成功分类: {class_name} -> {tool_dict['tool']}")
            except Exception as e:
                print(f"解析 {class_name} 响应失败: {e}")
        else:
            print(f"{class_name} 未获取到有效响应")

    print(f"===== 项目 {project_name} 处理完成，共成功分类 {len(results)} 个类 =====")
    return results, project_name


# 主函数
def main(output_excel, model="gpt-3.5-turbo"):
    all_results = []

    for config in PROJECT_CONFIGS:
        results, project_name = process_project(config, model)
        if results:
            all_results.append((results, project_name))

    if all_results:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            for results, project_name in all_results:
                df = pd.DataFrame(results, columns=["Class Name", "Suitable Tool"])
                sheet_name = project_name[:31]  # 限制 Excel 工作表名称长度
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"已保存 {project_name} 到工作表: {sheet_name}")

        print(f"\n所有结果已保存至: {os.path.abspath(output_excel)}")
    else:
        print("没有有效结果可保存")


if __name__ == "__main__":
    output_excel = "classification_results_all_7projects.xlsx"
    model = "gpt-3.5-turbo"

    # 验证项目数量
    if len(PROJECT_CONFIGS) != 7:
        print(f"警告：当前配置了 {len(PROJECT_CONFIGS)} 个项目，不是预期的 7 个")
    else:
        print("开始处理 7 个项目...")

    main(output_excel, model)