"""OpenAI API 后端"""

import json
import logging
import os
import re
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("evolution")

OPENAI_BASE_URL = "https://api.openai.com/v1"

_client: openai.OpenAI = None  # type: ignore
_custom_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, max_retries=0)


@once
def _setup_custom_client():
    global _custom_client
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        _custom_client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    查询 OpenAI API，可选支持函数调用。
    如果模型不支持函数调用，优雅降级为纯文本生成。
    """
    _setup_openai_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    if "max_tokens" in filtered_kwargs:
        filtered_kwargs["max_output_tokens"] = filtered_kwargs.pop("max_tokens")

    if (
        re.match(r"^o\d", filtered_kwargs["model"])
        or filtered_kwargs["model"] == "codex-mini-latest"
    ):
        filtered_kwargs.pop("temperature", None)

    model_name = filtered_kwargs.get("model", "")
    is_openai_model = re.match(r"^(gpt-|o\d-|codex-mini-latest$)", model_name)
    use_chat_api = os.getenv("OPENAI_BASE_URL") is not None and not is_openai_model

    if use_chat_api:
        _setup_custom_client()
        messages = opt_messages_to_list(system_message, user_message)
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
    else:
        messages = opt_messages_to_list(system_message, user_message)
        for i in range(len(messages)):
            messages[i]["content"] = [
                {"type": "input_text", "text": messages[i]["content"]}
            ]
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_responses_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_responses_tool_choice_dict

    logger.info(f"OpenAI API request: system={system_message}, user={user_message}")

    t0 = time.time()

    try:
        if use_chat_api:
            client_to_use = _custom_client if _custom_client else _client
            response = backoff_create(
                client_to_use.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            response = backoff_create(
                _client.responses.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                input=messages,
                **filtered_kwargs,
            )
    except openai.BadRequestError as e:
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            if use_chat_api:
                client_to_use = _custom_client if _custom_client else _client
                response = backoff_create(
                    client_to_use.chat.completions.create,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    **filtered_kwargs,
                )
            else:
                response = backoff_create(
                    _client.responses.create,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    input=messages,
                    **filtered_kwargs,
                )
        else:
            raise

    req_time = time.time() - t0

    if use_chat_api:
        message = response.choices[0].message

        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
            and func_spec is not None
        ):
            tool_call = message.tool_calls[0]
            if tool_call.function.name == func_spec.name:
                try:
                    output = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{tool_call.function.arguments}"
                    )
                    raise ex
            else:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {tool_call.function.name}. Fallback to text."
                )
                output = message.content
        else:
            output = message.content

        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
    else:
        if (
            hasattr(response, "output")
            and response.output is not None
            and len(response.output) > 0
        ):
            function_call_item = None
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    function_call_item = output_item
                    break

            if function_call_item is not None:
                if func_spec is not None and function_call_item.name == func_spec.name:
                    try:
                        output = json.loads(function_call_item.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(
                            "Error decoding function arguments:\n"
                            f"{function_call_item.arguments}"
                        )
                        raise ex
                else:
                    if func_spec is not None:
                        logger.warning(
                            f"Function name mismatch: expected {func_spec.name}, "
                            f"got {function_call_item.name}. Fallback to text."
                        )
                    output = response.output_text
            else:
                output = response.output_text
        else:
            output = response.output_text

        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    logger.info(
        f"OpenAI API call completed - {response.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"OpenAI API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
