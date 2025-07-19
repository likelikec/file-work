import os
import requests
import json
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from typing import List, Tuple

# 配置 API 密钥和 URL
API_KEY = "sk-BNBbQFIP7c4E04c33f86T3BlbKFJ4Ae550543E8b4e42b4b3"
API_URL = "https://cn2us02.opapi.win/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
}


@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    provider: str
    max_tokens: int = 2048
    temperature: float = 0.7
    supports_json_mode: bool = False


# 支持的模型配置
SUPPORTED_MODELS = {
    # OpenAI模型
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", "openai", 2048, 0.7, True),
    "gpt-4o-mini-2024-07-18": ModelConfig("gpt-4o-mini-2024-07-18", "openai", 12288, 0.5, True),

    # Anthropic Claude模型
    # "claude-3.7-sonnet": ModelConfig("claude-3-5-sonnet-20241022", "anthropic", 4096, 0.7, False),

    # Google Gemini模型
    "gemini-2.5-flash-lite-preview-06-17": ModelConfig("gemini-2.5-flash-lite-preview-06-17", "google", 12288, 0.5, False),

    # Qwen模型
    # "qwen-turbo": ModelConfig("qwen-turbo", "qwen", 2048, 0.7, False),
    # "qwen-plus": ModelConfig("qwen-plus", "qwen", 4096, 0.7, False),
    # "qwen-max": ModelConfig("qwen-max", "qwen", 4096, 0.7, False),
    # "qwen2-72b": ModelConfig("qwen2-72b-instruct", "qwen", 4096, 0.7, False),

    # 其他模型可以继续添加
}


def get_available_models() -> List[str]:
    """获取可用模型列表"""
    return list(SUPPORTED_MODELS.keys())


def print_available_models():
    """打印可用模型列表"""
    print("可用模型列表:")
    for provider_group in ["openai", "anthropic", "google", "qwen"]:
        models = [name for name, config in SUPPORTED_MODELS.items() if config.provider == provider_group]
        if models:
            provider_name = {
                "openai": "OpenAI",
                "anthropic": "Anthropic Claude",
                "google": "Google Gemini",
                "qwen": "Qwen"
            }.get(provider_group, provider_group)
            print(f"\n{provider_name}:")
            for model in models:
                print(f"  - {model}")


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


# 构造提示
def construct_prompt(class_name: str, class_code: str) -> str:
    prompt = f"""
Here is a Java class named {class_name}:

{class_code}
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


Please respond in the following JSON format:

{{"class_name": "{class_name}", "tool": "LLM" 或 "Evosuite"}}
"""
    return prompt


def construct_api_params(prompt: str, model_name: str) -> Dict[str, Any]:
    """根据模型构造API参数"""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}")

    config = SUPPORTED_MODELS[model_name]

    # 基础参数
    params = {
        "model": config.name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": False
    }

    # 如果模型支持JSON模式，添加响应格式
    if config.supports_json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


def parse_response(response_json: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    """解析不同模型的响应"""
    try:
        config = SUPPORTED_MODELS[model_name]

        # 获取响应内容
        if config.provider in ["openai", "qwen"]:
            content = response_json["choices"][0]["message"]["content"]
        elif config.provider == "anthropic":
            content = response_json["content"][0]["text"] if "content" in response_json else \
            response_json["choices"][0]["message"]["content"]
        elif config.provider == "google":
            content = response_json["candidates"][0]["content"]["parts"][0][
                "text"] if "candidates" in response_json else response_json["choices"][0]["message"]["content"]
        else:
            # 默认处理方式
            content = response_json["choices"][0]["message"]["content"]

        # 尝试解析JSON
        if content.strip().startswith("{"):
            return json.loads(content)
        else:
            # 如果不是JSON格式，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"无法解析响应内容: {content}")
                return None

    except Exception as e:
        print(f"解析响应时出错: {e}")
        return None


# 调用 API，添加重试机制
def call_api(prompt: str, model: str = "gpt-3.5-turbo", max_retries: int = 4) -> Optional[Dict[str, Any]]:
    """调用API，支持多种模型"""
    if model not in SUPPORTED_MODELS:
        print(f"错误: 不支持的模型 '{model}'")
        print_available_models()
        return None

    for attempt in range(max_retries):
        try:
            params = construct_api_params(prompt, model)
            print(f"[DEBUG] 正在使用模型 {model} 发送请求 (尝试 {attempt + 1}/{max_retries})...")

            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=params,
                timeout=30
            )
            response.raise_for_status()  # 检查 HTTP 状态码
            print("[DEBUG] 请求成功！")

            res_json = response.json()
            tool_dict = parse_response(res_json, model)

            if tool_dict:
                return tool_dict
            else:
                print(f"解析响应失败，尝试重试...")

        except requests.exceptions.RequestException as e:
            print(f"网络请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(2)  # 等待 2 秒后重试

    print(f"所有重试均失败，跳过此次请求")
    return None


# 处理单个项目
def process_project(project_dir: str, model: str = "gpt-3.5-turbo") -> Tuple[List[List[str]], str]:
    java_files = get_java_files(project_dir)
    results = []

    print(f"找到 {len(java_files)} 个Java文件")

    for i, file_path in enumerate(java_files, 1):
        class_name = get_class_name(file_path)
        print(f"处理文件 {i}/{len(java_files)}: {class_name}")

        try:
            class_code = read_file_content(file_path)
            prompt = construct_prompt(class_name, class_code)
            response = call_api(prompt, model)

            if response:
                try:
                    results.append([response["class_name"], response["tool"]])
                    print(f"  ✓ {class_name} -> {response['tool']}")
                except KeyError as e:
                    print(f"  ✗ 解析 {class_name} 的响应时缺少字段: {e}")
                except Exception as e:
                    print(f"  ✗ 解析 {class_name} 的响应时出错: {e}")
            else:
                print(f"  ✗ 跳过 {class_name}，继续处理下一个文件")
        except Exception as e:
            print(f"  ✗ 处理文件 {class_name} 时出错: {e}")

    # 返回结果和项目名
    project_name = os.path.basename(os.path.normpath(project_dir))
    return results, project_name


def test_model_call(model_name: str = "gpt-3.5-turbo") -> bool:
    """测试模型调用是否正常工作"""
    print(f"\n{'=' * 50}")
    print(f"🧪 测试模型调用: {model_name}")
    print(f"{'=' * 50}")

    # 验证模型是否支持
    if model_name not in SUPPORTED_MODELS:
        print(f"❌ 错误: 不支持的模型 '{model_name}'")
        print_available_models()
        return False

    # 构造测试提示
    test_class_name = "TestClass"
    test_class_code = """
public class TestClass {
    private int value;

    public TestClass(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public int add(int num) {
        return value + num;
    }
}
"""

    test_prompt = construct_prompt(test_class_name, test_class_code)

    print(f"📤 发送测试请求...")
    print(f"   模型: {model_name}")
    print(f"   配置: {SUPPORTED_MODELS[model_name]}")

    try:
        # 调用API
        response = call_api(test_prompt, model_name, max_retries=2)

        if response is None:
            print(f"❌ 测试失败: API调用返回None")
            return False

        # 验证响应格式
        if not isinstance(response, dict):
            print(f"❌ 测试失败: 响应不是字典格式")
            return False

        # 检查必要字段
        if "class_name" not in response:
            print(f"❌ 测试失败: 响应缺少 'class_name' 字段")
            print(f"   实际响应: {response}")
            return False

        if "tool" not in response:
            print(f"❌ 测试失败: 响应缺少 'tool' 字段")
            print(f"   实际响应: {response}")
            return False

        # 验证字段值
        if response["class_name"] != test_class_name:
            print(f"⚠️  警告: class_name不匹配")
            print(f"   期望: {test_class_name}")
            print(f"   实际: {response['class_name']}")

        if response["tool"] not in ["LLM", "Evosuite"]:
            print(f"⚠️  警告: tool值不在期望范围内")
            print(f"   期望: 'LLM' 或 'Evosuite'")
            print(f"   实际: {response['tool']}")

        # 测试成功
        print(f"✅ 测试成功!")
        print(f"   📋 响应内容:")
        print(f"      类名: {response['class_name']}")
        print(f"      推荐工具: {response['tool']}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def run_model_tests(models_to_test: List[str] = None) -> Dict[str, bool]:
    """运行多个模型的测试"""
    if models_to_test is None:
        # 自动使用SUPPORTED_MODELS中所有可用的模型
        models_to_test = list(SUPPORTED_MODELS.keys())
        print(f"📋 自动检测到以下可用模型:")
        for model in models_to_test:
            print(f"   - {model}")

    print(f"\n🚀 开始批量测试模型...")
    print(f"   测试模型数量: {len(models_to_test)}")

    results = {}
    success_count = 0

    for i, model in enumerate(models_to_test, 1):
        print(f"\n📊 进度: {i}/{len(models_to_test)}")

        if model not in SUPPORTED_MODELS:
            print(f"⏭️  跳过不支持的模型: {model}")
            results[model] = False
            continue

        try:
            success = test_model_call(model)
            results[model] = success
            if success:
                success_count += 1

            # 测试间隔，避免请求过快
            if i < len(models_to_test):
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n⏹️  用户中断测试")
            break
        except Exception as e:
            print(f"❌ 测试模型 {model} 时出现异常: {e}")
            results[model] = False

    # 显示测试总结
    print(f"\n{'=' * 60}")
    print(f"📊 测试总结")
    print(f"{'=' * 60}")
    print(f"总测试数量: {len(results)}")
    print(f"成功数量: {success_count}")
    print(f"失败数量: {len(results) - success_count}")
    print(f"成功率: {success_count / len(results) * 100:.1f}%")

    print(f"\n📋 详细结果:")
    for model, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {model:<30} {status}")

    return results


# 主函数
def main(project_dirs: List[str], output_excel: str, model: str = "gpt-3.5-turbo", test_mode: bool = False):
    # 如果是测试模式，只运行测试
    if test_mode:
        print("🧪 运行测试模式...")
        success = test_model_call(model)
        if success:
            print(f"\n✅ 模型 {model} 测试通过，可以正常使用！")
        else:
            print(f"\n❌ 模型 {model} 测试失败，请检查配置！")
        return

    # 验证模型
    if model not in SUPPORTED_MODELS:
        print(f"错误: 不支持的模型 '{model}'")
        print_available_models()
        return

    # 运行测试确认模型可用
    print(f"🔍 首先测试模型 {model} 是否可用...")
    if not test_model_call(model):
        print(f"❌ 模型测试失败，程序终止")
        return

    print(f"\n✅ 模型测试通过，开始处理项目...")
    print(f"使用模型: {model}")
    print(f"输出文件: {output_excel}")

    # 创建一个 ExcelWriter 对象
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 处理每个项目
        for project_dir in project_dirs:
            print(f"\n{'=' * 60}")
            print(f"开始处理项目: {project_dir}")
            print(f"{'=' * 60}")

            results, project_name = process_project(project_dir, model)

            # 保存结果到对应的 sheet
            if results:
                df = pd.DataFrame(results, columns=["Class Name", "Suitable Tool"])
                df.to_excel(writer, sheet_name=project_name[:31], index=False)  # sheet 名称不能超过31个字符
                print(f"\n✅ 项目 {project_name} 的结果已保存到 {output_excel} 的 sheet: {project_name}")
                print(f"  共处理 {len(results)} 个类")
            else:
                print(f"\n❌ 项目 {project_name} 没有结果可保存")


if __name__ == "__main__":
    # 打印可用模型
    print_available_models()

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

    output_excel = "gemini-14-classification_results.xlsx"

    # 可以轻松切换模型
    model = "gemini-2.5-flash-lite-preview-06-17"  # 可以改为: "claude-3.5-sonnet", "gemini-1.5-pro", "qwen-plus" 等

    # 使用选项
    # 1. 测试单个模型: 设置 TEST_MODE = True
    # 2. 批量测试多个模型: 设置 RUN_BATCH_TEST = True
    # 3. 正常运行: 两个都设置为 False
    TEST_MODE = False  # 设置为 True 只运行测试
    RUN_BATCH_TEST = False  # 设置为 True 运行批量模型测试

    if RUN_BATCH_TEST:
        # 批量测试多个模型
        run_model_tests()  # 不传参数，自动使用SUPPORTED_MODELS中的所有模型
    else:
        # 正常运行或单个模型测试
        main(project_dirs, output_excel, model, TEST_MODE)