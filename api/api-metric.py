import os
import requests
import json
import pandas as pd
from datetime import datetime

# 配置 API 密钥和 URL（根据示例代码调整）
API_KEY = "sk-BNBbQFIP7c4E04c33f86T3BlbKFJ4Ae550543E8b4e42b4b3"
API_URL = "https://c-z0-api-01.hash070.com/v1/chat/completions"  # 直接使用完整 URL

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",  # 示例代码中的格式
}


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


# 从项目路径提取项目名
def get_project_name(project_path):
    # 获取最后一个文件夹名作为项目名
    return os.path.basename(os.path.normpath(project_path))


# 读取文件内容
def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 构造提示（保持不变）
def construct_prompt(class_name, class_code):
    prompt = f"""
Here is a Java class named {class_name}:

{class_code}
You are a senior professor of software engineering. Please classify each class based on your professional knowledge and the following 25 indicators that describe code characteristics. Read the code, combine the characteristics of test case generation by Evo and LLM, calculate the 25 indicators that describe code characteristics to determine whether it is more appropriate to use LLM or Evosuite to generate test cases.

Evosuite is a tool that automatically generates Java-like test cases using evolutionary algorithms, with the goal of achieving high code coverage. LLM can generate test cases based on its understanding of the code and behavior.

The 25 indicators are as follows:

1. Maximum Cyclomatic Complexity (OCmax) : It is a logical complexity metric, referring to the maximum cyclomatic complexity of all non-abstract methods in a class. The basic complexity of each method is 1, and when encountering a specific control flow structure, the complexity increases by 1, reflecting the number of branches in the code and the difficulty of understanding the methods.
2. **Average Operation Complexity (OCavg)** : Derived from OCmax, it refers to the average of the cyclomatic complexity of all non-abstract methods in a class, reflecting the average logical complexity of the method set in the class.
3. **Weighted Methods per Class (WMC)** : It is a logical complexity metric, referring to the sum of the complexities of all methods in a class, comprehensively reflecting the total logical complexity of all methods in the class and its close relationship with the overall logical complexity of the class method set.
4. **Number of Dependencies (Dcy)** : It is an object-oriented coupling metric, referring to the number of other classes that a class depends on. It is calculated by constructing a dependency graph and using graph traversal algorithms and is negatively correlated with the degree of modularization of the system.
5. **Number of Transitive Dependencies (Dcy*)** : Derived from Dcy, it refers to the number of other classes that a class directly or indirectly depends on, including the number of directly dependent classes and indirectly dependent classes, which can reflect the deep coupling state of the class in the dependency network.
6. **Number of Dependents (DPT)** : It is an object-oriented coupling metric, referring to the number of other classes that directly depend on the current class. It is obtained by searching for other classes that use the current class in the project and counting the reference results.
7. **Number of Transitive Dependents (DPT*)** : It is an object-oriented coupling feature, referring to the number of classes that indirectly depend on the current class through the dependency chain. It is obtained by constructing a dependency graph and using a graph traversal algorithm to find all transitive dependencies and then statistically analyzing them.
8. **Cyclic Dependencies ** : It is an object-oriented coupling feature, referring to the number of other classes that a class directly or indirectly depends on, and these dependencies also directly or indirectly depend on that class. It is obtained by constructing a dependency graph, collecting direct dependencies, calculating transitive dependencies, identifying strongly connected components, and then subtracting 1 after statistics, representing the number of other classes involved in circular dependencies.
9. **Level (Level)** : It is an object-oriented theoretical complexity feature that measures the "number of levels" of the classes a class depends on. When it does not depend on other classes, its value is 0; when it does depend, its value is the maximum Level value among the dependent classes plus 1 (excluding classes that are mutually dependent or cyclically dependent).
10. Adjusted Level (Level*) : It is a theoretical complexity feature of object-oriented programming. Based on the "number of layers" of the classes that a class depends on, it considers the number of classes in circular dependencies. When it does not depend on other classes, the value is 0. When there is a dependency, the value is the maximum Level* value in the dependent class (non-circular dependency) plus the number of classes that are interdependent with or form a circular dependency with that class.
11. **Package Dependency Count (PDcy)** : Extended from Robert C. Martin's package-level dependency metric "outgoing coupling (Ce)" to the class level, it refers to the number of external packages that a class directly depends on, reflecting the degree of coupling between the class and external modules.
12. **Transitive Package Dependency Count (PDcy*)** : Introduced on the basis of PDcy, considering the problem of circular dependency, it refers to the number of packages that a class directly and indirectly depends on, which is closely related to the code structure and the degree of modularization.
13. **Package Dependents (PDpt)** : It is the dependency feature corresponding to PDcy, referring to the number of packages that directly or indirectly depend on the current class. It is obtained by counting the number of packages that directly or indirectly depend on the class.
14. **Depth of Inheritance Tree (DIT)** : It is an object-oriented measure of inheritance depth, referring to the depth of the inheritance tree for each class. It is calculated by recursively tracing the inheritance chain from the current class to the root class. The depth value of each parent class is increased by 1, and 0 is returned when there is no parent class. It is related to the complexity of software design.
15. Coupling Between Objects (CBO) : It is an object-oriented coupling metric, referring to the degree of "coupling" between one class and other classes. When one class depends on another class or is dependent on another class, the two are coupled. Its value is the size of the union of the set of classes that depend on the current class and the set of classes that the current class depends on.
16. **Message Passing Coupling (MPC)** : A quantification method of message passing coupling proposed by Briand, it refers to the number of messages (method calls) sent by one class to other classes, excluding the calls to methods of the same class, and measures the degree of coupling between classes from the perspective of dynamic interaction frequency.
17. Lack of Cohesion of Methods (LCOM) : It is an object-oriented cohesion feature, referring to the degree of lack of cohesion among the methods of a class. It is obtained by constructing an inter-method relationship graph (where nodes represent methods and edges represent inter-method relationships) and calculating the number of connected components in the graph, and is related to the cohesion and responsibility of the class.
18. **Response for Class (RFC)** : It refers to the size of the response set of a class, including the methods within the class and the external methods that are called. It is obtained by counting all the methods defined in the class and all the external methods that are called. The higher the value, the more complex the behavior of the class.
19. **Number of Queries (Query)** : Quantifies the ability of a class to provide information query services, referring to the number of non-constructor methods (excluding inherited methods) with return values in the class, which is related to the design complexity of the code and the allocation of responsibilities.
20. **Comment Lines of Code (CLOC)** : It is an extension of the concept of source code lines of code (SLOC) proposed by Lazic. It refers to the total number of lines containing comment content in the code file, calculated by strict syntax parsing, reflecting the physical density of comments in the code and its relationship with the interpretability of the code.
21. **Javadoc Lines of Code (JLOC)** : It is a refined metric of CLOC, specifically counting the number of lines of comments that comply with the Javadoc specification, measuring the density of code comments that follow the conventions of the standard API documentation, and is closely related to the completeness of the API documentation.
22. **Javadoc Field Coverage (JF)** : It is a coverage feature based on JLOC, referring to the percentage of the number of fields with Javadoc comments to the total number of fields in the class, quantifying the documentation level of fields in the class.
23. **Javadoc Method Coverage (JM)** : Derived from JLOC, it refers to the percentage of the number of methods with Javadoc annotations to the total number of methods in the class, and is closely related to the integrity of the method-level documentation.
24. **Comment Ratio (COM_RAT)** : It is a code comment feature, referring to the ratio of the number of comment lines in the code to the total number of code lines (excluding blank lines), reflecting the relative density of comments in the code. It is a classic feature for evaluating code readability.
25. True Comment Ratio (TCOM_RAT) : It is a code comment quality metric, referring to the ratio of the number of valid comment lines (excluding automatically generated comments, only including meaningful document comments or explanatory comments) to the total number of code lines (excluding blank lines), reflecting the quality density of valid comments in the code and can be used to evaluate code readability and documentation integrity.
26. **Number of TODO Comments (TODO)** : It is a code comment metric, referring to the total number of TODO comments in a class.
27. **Lines of Code (LOC)** : It is a metric, referring to the total number of lines of code in a class, including comment lines but excluding pure blank lines. It is a fundamental indicator for measuring the size of code.
28. **Non-Comment Lines of Code (NCLOC)** : It is a derivative metric of LOC, referring to LOC minus the number of pure comment lines, eliminating the interference of comments and accurately reflecting the scale of the actual executable code.
29. **Class Size Attributes (CSA)** : Proposed based on the research of Li & Henry, it refers to the number of non-static attributes defined in a class.
30. **Class Size Operations (CSO)** : As a dual metric of CSA, it refers to the number of non-static methods defined in a class.
31. **Class Size (Operations + Attributes) (CSOA)** : It is a measure of the logical size of a class, referring to the total number of operations and attributes in the class, that is, the sum of CSA and CSO.
32. **Maximum Operation Size (OSmax)** : It is a measure of the structural size of a class, referring to the number of statements contained in the maximum method in the class.
33. **Average Operation Size (OSavg)** : It is a derivative metric of OPavg, referring to the average number of statements for each method in the class, which is calculated by dividing the sum of the number of statements of all methods by the total number of methods.
34. **Average Number of Parameters (OPavg)** : It is a measure of the structural scale of a class, referring to the average number of parameters for each method in the class, which is closely related to the complexity of the method interface. It is calculated by dividing the sum of the parameters of all methods by the total number of methods.
35. **Number of Operations Added (NOAC)** : It is a measure of the structural scale of a class, referring to the number of new methods added by the subclass. It only counts the newly defined methods in the class and does not include methods inherited from the parent class or overridden methods.
36. **Number of Attributes Added (NAAC)** : As a dual indicator of NOAC, it refers to the number of new attributes added to a class. It is obtained by first counting the attributes directly defined in the class and then excluding those inherited from the parent class.
37. **Number of Operations Overridden (NOOC)** : This is a measure of the logical scale of a class, referring to the number of methods overridden by a subclass in its parent class. It is obtained by checking whether each method in the class uses the @Override annotation or overrides the methods of the parent class.
38. **Number of Attributes Inherited (NAIC)** : It is a derivative indicator of NOOC, referring to the number of attributes inherited by a class from its parent class. It is obtained by traversing the inheritance hierarchy of the class, counting all accessible attributes inherited from the parent class, and excluding the attributes defined by the current class.
39. Number of Operations Inherited (NOIC) : It is a derivative metric of NOAC, referring to the total number of operations (or methods) inherited by a class. The remaining quantity is obtained by traversing all the methods of the class, excluding constructors, private methods, static methods, abstract methods, and methods defined by the current class itself.
40. **Number of Children (NOC)** : It is an object-oriented measure of the structural scale of a class, referring to the total number of direct subclasses of a class. It helps assess the inheritance hierarchy and reuse of a class. A high value indicates that the class is widely inherited and may be a well-designed base class/superclass.
41. **Number of Constructors (CONS)** : It refers to the total number of constructors declared in a class, only counting those explicitly declared in the class. If no constructors are declared, the value is 0.
42. **Number of Implemented Interfaces (INNER)** : Defined considering the need to describe the size of a class from the perspective of interface implementation, it refers to the total number of interfaces implemented by a class, including all different interfaces implemented directly and inherited from the parent class.
43. **Number of Inner Classes (Inner)** : It is a derivative metric of NOAC, referring to the number of inner classes or interfaces contained in a class.
44. **Number of Type Parameters (NTP)** : It is a measure of the structural size of a class, referring to the number of type parameters in a class. A high value indicates that the class uses more type parameters, which may lead to an increase in the class size.
45. **Number of Statements (STAT)** : It refers to the number of valid statements in a method. It is obtained by traversing all PSI statement nodes, excluding empty statements and block statements, and then counting the actual executable statements.
46. **String Processing** : This scenario involves the creation, manipulation, parsing, formatting or transformation of strings, such as processing text data, generating reports, cleaning input, as well as methods like string splitting, concatenation, and regular expression matching.
47. **File Operations** : This scenario involves interaction with the file system and operations on file contents, specifically including file creation, reading, and parsing and management of file paths, such as reading configuration files and exporting data as CSV, etc.
48. **Network Communication** : This scenario involves the management of network requests, responses, data transmission or protocol processing, such as calling apis, implementing WebSocket communication or handling HTTP requests, etc.
49. **Database Operations** : This scenario mainly deals with database connections, queries, updates or transaction management, such as executing SQL queries, operating ORM, managing data storage or defining data models, etc.
50. Mathematical Computation: This scenario mainly involves numerical computation, algorithm implementation, or statistical analysis, including mathematical functions, statistical models, or optimization algorithms, such as handling financial calculations, machine learning predictions, or geometric operations, etc.
51. **User Interface** : This scenario manages user interaction, interface rendering or event handling, such as building Web front ends, desktop GUIs or mobile interfaces, etc.
52. **Business Logic** : This scenario mainly implements specific business rules or domain logic, which is specific to the application context, such as user permission verification, workflow engine, etc.
53. **Data Structures and Algorithms** : This scenario manages data sets and achieves efficient storage or manipulation, such as processing lists, trees, graphs, traversing data structures or optimizing methods (such as sorting, searching), etc.
54. **System and Tools** : This scenario mainly provides general auxiliary functions or system-level operations, such as logging, error handling, or process management, etc. Other categories usually reuse the code of this category.
55. Concurrency and Multithreading: This scenario mainly deals with parallel execution, thread management, or resource sharing, such as implementing asynchronous tasks or avoiding race conditions, etc.
56. **Exception Handling** : This scenario is responsible for uniformly managing the system's exception handling mechanism, including exception capture, classification, handling, reporting, and the implementation of exception recovery strategies, etc.
57. **Halstead's incorrect prediction (B) ** : Derived from Halstead's related works, it is a metric used to predict the number of possible defects in software code, with the calculation formula being B = Math.pow(effort, 2.0/3.0) / 3000.0 (effort is Halstead's workload metric) The theoretical predicted value (calculated by multiplying the difficulty coefficient by the capacity) usually underestimates the actual number of errors in practical applications.
58. **Halstead Length (N) ** : Derived from related works on Halstead, it refers to the total length of the program, that is, the total number of operators and operands. The calculation formula is N = N1 + N2 (N1 is the total number of operators, and N2 is the total number of operands).
59. **Halstead Vocabulary (n) ** : Derived from Halstead's related works, it refers to the total number of different operators and operands in a program, with the calculation formula being n = n1 + n2 (n1 represents the number of different operators, and n2 represents the number of different operands).
60. **Halstead Difficulty Coefficient (D) ** : Derived from Halstead's related works, it refers to the difficulty of understanding and maintaining a program. The calculation formula is D = (n1/2) × (N2/n2) (n1 represents the number of different operators, N2 represents the total number of operands, and n2 represents the number of different operands).
61. **Halstead Capacity (V) ** : Derived from Halstead's related works, it refers to the information capacity of a program, indicating the amount of information contained in the program. The calculation formula is V = N × log2(N) (where N is the program length and n is the vocabulary).
62. **Halstead Workload (E) ** : Derived from Halstead's related works, it refers to the mental workload required to understand a program. The calculation formula is E = D × V (where D is the difficulty coefficient and V is the capacity).



Please respond in the following JSON format:

{{"class_name": "{class_name}", "tool": "LLM" or "Evosuite"}}
"""
    return prompt


# 调用 API（根据示例代码调整参数结构）
def call_api(prompt, model="gpt-3.5-turbo"):
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "stream": False  # 示例代码中的 stream=False
    }

    try:
        # 添加超时和调试信息
        print("[DEBUG] 正在发送请求...")
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=params,
            timeout=30  # 设置超时
        )
        response.raise_for_status()  # 检查 HTTP 状态码

        print("[DEBUG] 请求成功！")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 调用失败: {e}")
        return None


# 处理单个项目
def process_project(project_dir, model="gpt-3.5-turbo"):
    java_files = get_java_files(project_dir)
    results = []
    project_name = get_project_name(project_dir)

    print(f"\n正在处理项目: {project_name}")
    print(f"找到 {len(java_files)} 个 Java 文件")

    for i, file_path in enumerate(java_files, 1):
        class_name = get_class_name(file_path)
        print(f"正在处理文件 {i}/{len(java_files)}: {class_name}")

        class_code = read_file_content(file_path)
        prompt = construct_prompt(class_name, class_code)
        response = call_api(prompt, model)

        if response:
            try:
                # 解析响应内容
                res_content = response["choices"][0]["message"]["content"]
                tool_dict = json.loads(res_content)
                results.append([tool_dict["class_name"], tool_dict["tool"]])
            except Exception as e:
                print(f"解析 {class_name} 的响应时出错: {e}")
        else:
            print(f"获取 {class_name} 的响应失败")

    return results, project_name


# 主函数
def main(project_dirs, output_excel, model="gpt-4o-mini-2024-07-18"):
    # 创建 ExcelWriter 对象
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 遍历处理每个项目
        for project_dir in project_dirs:
            results, project_name = process_project(project_dir, model)

            # 保存结果到 Excel 的不同 sheet
            if results:
                df = pd.DataFrame(results, columns=["Class Name", "Suitable Tool"])
                # 限制 sheet 名称长度为 31 个字符（Excel 限制）
                sheet_name = project_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"项目 '{project_name}' 的结果已保存到 sheet: {sheet_name}")
            else:
                print(f"项目 '{project_name}' 没有结果可保存")

    print(f"\n所有结果已保存到 {output_excel}")


if __name__ == "__main__":
    # 定义7个项目的目录
    project_dirs = [

                "E:\\unit-generate\\commons-csv\\src\\main\\java\\org\\apache\\commons\\csv",
                "E:\\unit-generate\\commons-lang\\src\\main\\java\\org\\apache\\commons\\lang3",
                "E:\\unit-generate\\google-json\\src\\main\\java\\com\\google\\gson",
                "E:\\unit-generate\\commons-cli-evo\\src\\main\\java\\org\\apache\\commons\\cli",
                "E:\\unit-generate\\ruler\\src\\main\\software\\amazon\\event\\ruler",
                "D:\\restful-demo-1\\dat\\src\\main\\java\\net\\datafaker",
        "E:\\unit-generate\\jfreechart154\\src\\main\\java\\org\\jfree"
    ]

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = f"metric-classification_results_{timestamp}.xlsx"

    model = "gpt-4o-mini-2024-07-18"  # 可切换模型
    main(project_dirs, output_excel, model)