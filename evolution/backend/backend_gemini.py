"""Gemini API 后端（使用 OpenAI 兼容接口）"""

import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("evolution")

_client: openai.OpenAI = None  # type: ignore

GEMINI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_gemini_client():
    global _client
    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key = os.getenv("GEMINI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=gemini_base_url, max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    通过 OpenAI 兼容接口查询 Gemini API，可选支持函数调用。
    """
    _setup_gemini_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.info(f"Gemini API request: system={system_message}, user={user_message}")

    completion = None
    t0 = time.time()

    try:
        completion = backoff_create(
            _client.chat.completions.create,
            GEMINI_TIMEOUT_EXCEPTIONS,
            messages=messages,
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

            completion = backoff_create(
                _client.chat.completions.create,
                GEMINI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            raise

    req_time = time.time() - t0
    choice = completion.choices[0]

    if func_spec is None or "tools" not in filtered_kwargs:
        output = choice.message.content
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            logger.warning(
                "No function call was used despite function spec. Fallback to text.\n"
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            if first_call.function.name != func_spec.name:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {first_call.function.name}. Fallback to text."
                )
                output = choice.message.content
            else:
                try:
                    output = json.loads(first_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{first_call.function.arguments}"
                    )
                    raise ex

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": completion.model,
        "created": getattr(completion, "created", None),
    }

    logger.info(
        f"Gemini API call completed - {completion.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"Gemini API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
