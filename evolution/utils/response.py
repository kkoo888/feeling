"""响应解析工具：从 LLM 输出中提取代码、文本等"""

import json
import re

try:
    import black
    HAS_BLACK = True
except ImportError:
    HAS_BLACK = False


def wrap_code(code: str, lang="python") -> str:
    """用三个反引号包裹代码"""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """检查脚本是否为有效的 Python 脚本"""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """从文本中提取所有 JSON 对象（不支持嵌套）"""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    """截断过长的字符串，保留首尾各 k 个字符"""
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """从文本中提取 Python 代码块"""
    parsed_codes = []

    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """提取第一个代码块之前的自然语言文本"""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """使用 Black 格式化 Python 代码（如果可用）"""
    if not HAS_BLACK:
        return code
    try:
        return black.format_str(code, mode=black.FileMode())
    except (black.parsing.InvalidInput, Exception):  # type: ignore
        return code
