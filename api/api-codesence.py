import os
import requests
import json
import pandas as pd

# 配置 API 密钥和 URL（根据示例代码调整）
API_KEY = "sk-BNBbQFIP7c4E04c33f86T3BlbKFJ4Ae550543E8b4e42b4b3"
API_URL = "https://c-z0-api-01.hash070.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
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


# 读取文件内容
def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 构造提示（保持不变）
def construct_prompt(class_name, class_code):
    prompt = f"""

You are a senior professor of software engineering. 
Please classify the functions of each category based on your professional knowledge and the following 11 indicators that describe the responsibilities of the code. 
Read the code and understand its functions. Each class may have more than one responsibility, if the responsibility is included, fill in "yes"; if not, write "no".

These 11 responsibilities are as follows:

1. String processing

Problems solved: Involving the creation, operation, parsing, formatting or conversion of strings. For example, processing text data, generating reports or cleaning inputs.

Typical code: It includes methods such as string splitting, concatenation, replacement, and regular expression matching.

2. File operations

The problems solved: Handle the reading, writing, creation, deletion, path management or file content parsing of the file system. For example, read configuration files and export data to CSV.

Typical code: including file read and write functions, directory traversal or file format processing (such as JSON/XML parsing).

3. Network communication

The problem to be solved: Managing network requests, responses, data transmission or protocol processing. For example, calling apis, implementing WebSocket communication or handling HTTP requests.

Typical code: involves HTTP client/server implementation, socket operations, or RESTful API calls.

4. Database operations

Problems solved: Handling database connections, queries, updates or transaction management. For example, perform SQL queries, operate ORM (Object-Relational Mapping), or manage data stores.

Typical code: including SQL execution methods, connection pool management or data model definition.

5. Mathematical calculation

Problems to be solved: Performing numerical calculations, algorithm implementation or statistical analysis. For example, handling financial calculations, machine learning predictions or geometric operations.

Typical code: Involving mathematical functions, statistical models or optimization algorithms.

6. User Interface (UI)

Problems to be solved: Managing user interaction, interface rendering or event handling. For example, build Web front-end, desktop GUI or mobile interface.

Typical code: including UI component rendering, event listening or state management.

7. Business Logic

The problem to be solved: implementing specific business rules or domain logic related to specific applications. For example, e-commerce order processing, user permission verification or workflow engines.

Typical code: Encapsulate business processes, such as validation rules, decision trees, or service coordination.

8. Data Structures and Algorithms

The problem to be solved: managing data sets, achieving efficient storage or operation. For example, handle lists, trees, graphs or caching mechanisms.

Typical code: including data structure initialization, traversal or optimization methods (such as sorting, searching).

9. Systems and Tools

The problem to be solved: Provide general auxiliary functions or system-level operations, such as logging, error handling or process management. These codes are usually reused by other categories.

Typical code: utility functions, tool methods or system calls.

10. Concurrency and Multithreading

The problems solved: handling parallel execution, thread management or resource sharing. For example, implementing asynchronous tasks or avoiding race conditions.

Typical code: including thread pool management, lock mechanism or concurrent data structure.

11. Exception handling

The exception handling mechanism of the unified management system.

Here is a Java class named {class_name}:{class_code}

Please respond in the following JSON format, :

{{"class_name": "{class_name}", "String processing": "yes" or "no","File operations": "yes" or "no","Network communication": "yes" or "no",
"Database operations": "yes" or "no",
"Mathematical calculation": "yes" or "no",
"User Interface (UI)": "yes" or "no",
"Business Logic": "yes" or "no",
"Data Structures and Algorithms": "yes" or "no",
"Systems and Tools": "yes" or "no",
"Concurrency and Multithreading": "yes" or "no",
"Exception handling": "yes" or "no"}}
"""
    return prompt


# 调用 API（根据示例代码调整参数结构）
def call_api(prompt, model="gemini-2.0-flash"):
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


# 主函数
def main(project_dir, output_excel, model="gemini-2.0-flash"):
    java_files = get_java_files(project_dir)

    # 定义11个责任类别的列名
    columns = ["class_name", "String processing", "File operations", "Network communication",
               "Database operations", "Mathematical calculation", "User Interface (UI)",
               "Business Logic", "Data Structures and Algorithms", "Systems and Tools",
               "Concurrency and Multithreading", "Exception handling"]

    results = []

    for file_path in java_files:
        class_name = get_class_name(file_path)
        class_code = read_file_content(file_path)
        prompt = construct_prompt(class_name, class_code)
        response = call_api(prompt, model)

        if response:
            try:
                # 解析响应内容
                res_content = response["choices"][0]["message"]["content"]
                # 清理可能的JSON格式问题（如首尾多余字符）
                res_content = res_content.strip()
                if res_content.startswith('```json') and res_content.endswith('```'):
                    res_content = res_content[8:-3].strip()

                tool_dict = json.loads(res_content)

                # 提取所有11个类别的结果
                result_row = [
                    tool_dict["class_name"],
                    tool_dict["String processing"],
                    tool_dict["File operations"],
                    tool_dict["Network communication"],
                    tool_dict["Database operations"],
                    tool_dict["Mathematical calculation"],
                    tool_dict["User Interface (UI)"],
                    tool_dict["Business Logic"],
                    tool_dict["Data Structures and Algorithms"],
                    tool_dict["Systems and Tools"],
                    tool_dict["Concurrency and Multithreading"],
                    tool_dict["Exception handling"]
                ]
                results.append(result_row)
            except Exception as e:
                print(f"解析 {class_name} 的响应时出错: {e}")
                # 解析失败时添加空行以便追踪
                results.append([class_name] + ["N/A"] * 11)
        else:
            print(f"获取 {class_name} 的响应失败")
            # 响应失败时添加空行以便追踪
            results.append([class_name] + ["Failed"] * 11)

    # 保存结果
    if results:
        df = pd.DataFrame(results, columns=columns)
        df.to_excel(output_excel, index=False)
        print(f"结果已保存到 {output_excel}")
        # 打印Excel文件的前几行以便检查
        print("结果预览:")
        print(df.head())
    else:
        print("没有结果可保存")


if __name__ == "__main__":
    project_dir = "E:\\unit-generate\\jfreechart154\\src\\main\\java\\org\\jfree"
    output_excel = "codersence.xlsx"
    model = "gemini-2.0-flash"  # 可切换模型
    main(project_dir, output_excel, model)