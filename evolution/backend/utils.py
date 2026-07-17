"""后端工具函数：通用的消息编译、重试逻辑和函数调用规范"""

from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin
import backoff
import logging
from typing import Callable

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


logger = logging.getLogger("evolution")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    """带指数退避的重试包装器"""
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    """将可选的系统/用户消息转换为消息列表"""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """将嵌套的 prompt 字典编译为 Markdown 格式"""
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    """函数调用规范，支持 OpenAI 和 Anthropic 格式"""
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # 验证 schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """转换为 OpenAI 函数格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """转换为 Anthropic 工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic 使用 input_schema 而非 parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """转换为 Anthropic 工具选择格式"""
        return {
            "type": "tool",  # Anthropic 使用 "tool" 而非 "function"
            "name": self.name,
        }

    @property
    def as_openai_responses_tool_dict(self):
        """转换为 OpenAI Responses API 工具格式"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_responses_tool_choice_dict(self):
        """转换为 OpenAI Responses API 工具选择格式"""
        return {
            "type": "function",
            "name": self.name,
        }
