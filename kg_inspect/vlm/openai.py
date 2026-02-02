from lightrag.utils import verbose_debug, VERBOSE_DEBUG
import os
import logging
import mimetypes
from collections.abc import AsyncIterator

import pipmaster as pm

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
)

from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__

import numpy as np
import base64
from typing import Any, List, Optional, Union
from lightrag.llm.openai import InvalidResponseError
from dotenv import load_dotenv

# Try to import Langfuse for LLM observability (optional)
# Falls back to standard OpenAI client if not available
# Langfuse requires proper configuration to work correctly
LANGFUSE_ENABLED = False
try:
    # Check if required Langfuse environment variables are set
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    # Only enable Langfuse if both keys are configured
    if langfuse_public_key and langfuse_secret_key:
        from langfuse.openai import AsyncOpenAI  # type: ignore[import-untyped]

        LANGFUSE_ENABLED = True
        logger.info("Langfuse observability enabled for OpenAI client")
    else:
        from openai import AsyncOpenAI

        logger.debug(
            "Langfuse environment variables not configured, using standard OpenAI client"
        )
except ImportError:
    from openai import AsyncOpenAI

    logger.debug("Langfuse not available, using standard OpenAI client")

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    if not api_key:
        api_key = os.environ["OPENAI_API_KEY"]

    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    if client_configs is None:
        client_configs = {}

    merged_configs = {
        **client_configs,
        "default_headers": default_headers,
        "api_key": api_key,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    return AsyncOpenAI(**merged_configs)


def _build_vision_user_content(prompt: str, images: Optional[List[str]] = None):
    if not images:
        return prompt

    content_blocks: List[dict] = []
    if prompt:
        content_blocks.append({"type": "text", "text": prompt})

    ok_image_count = 0

    for img_path in images:
        if not img_path:
            continue

        if img_path.startswith("http://") or img_path.startswith("https://"):
            content_blocks.append({"type": "image_url", "image_url": {"url": img_path}})
            ok_image_count += 1
            continue
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type:
            mime_type = "image/png"

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        ok_image_count += 1

    if ok_image_count == 0:
        raise InvalidResponseError("No images were successfully added to the request.")

    return content_blocks


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    images: Optional[List[str]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []

    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    kwargs.pop("hashing_kv", None)
    client_configs = kwargs.pop("openai_client_configs", {})

    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        client_configs=client_configs,
    )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)

    custom_messages = kwargs.pop("messages", None)
    if custom_messages is not None:
        messages = custom_messages
    else:
        user_content = _build_vision_user_content(prompt, images)
        messages.append({"role": "user", "content": user_content})
        
    logger.debug("===== Entering func of LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Client Configs: {client_configs}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    if stream is not None:
        kwargs["stream"] = stream
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except Exception:
        await openai_async_client.close()
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            final_chunk_usage = None
            cot_active = False
            cot_started = False
            initial_content_seen = False

            try:
                async for chunk in response:
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage

                    if not hasattr(chunk, "choices") or not chunk.choices:
                        continue
                    if not hasattr(chunk.choices[0], "delta"):
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning_content = getattr(delta, "reasoning_content", "")

                    if enable_cot:
                        if content:
                            if not initial_content_seen:
                                initial_content_seen = True
                                if reasoning_content:
                                    cot_active = False
                                    cot_started = False

                            if cot_active:
                                yield "</think>"
                                cot_active = False

                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                        elif reasoning_content:
                            if not initial_content_seen and not cot_started:
                                if not cot_active:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True

                            if cot_active:
                                if r"\u" in reasoning_content:
                                    reasoning_content = safe_unicode_decode(
                                        reasoning_content.encode("utf-8")
                                    )
                                yield reasoning_content
                    else:
                        if content:
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                if enable_cot and cot_active:
                    yield "</think>"
                    cot_active = False

                if token_tracker and final_chunk_usage:
                    token_counts = {
                        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(final_chunk_usage, "completion_tokens", 0),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)

            finally:
                if hasattr(response, "aclose"):
                    aclose_method = getattr(response, "aclose", None)
                    if callable(aclose_method):
                        try:
                            await response.aclose()
                        except Exception:
                            pass
                await openai_async_client.close()

        return inner()

    try:
        if not response or not response.choices or not hasattr(response.choices[0], "message"):
            raise InvalidResponseError("Invalid response from OpenAI API")

        message = response.choices[0].message
        content = getattr(message, "content", None)
        reasoning_content = getattr(message, "reasoning_content", "")

        final_content = ""

        if enable_cot:
            should_include_reasoning = False
            if reasoning_content and reasoning_content.strip():
                if not content or content.strip() == "":
                    should_include_reasoning = True
                    final_content = content or ""
                else:
                    should_include_reasoning = False
                    final_content = content
            else:
                final_content = content or ""

            if should_include_reasoning:
                if r"\u" in reasoning_content:
                    reasoning_content = safe_unicode_decode(reasoning_content.encode("utf-8"))
                final_content = f"<think>{reasoning_content}</think>{final_content}"
        else:
            final_content = content or ""

        if not final_content or final_content.strip() == "":
            raise InvalidResponseError("Received empty content from OpenAI API")

        if r"\u" in final_content:
            final_content = safe_unicode_decode(final_content.encode("utf-8"))

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        return final_content
    finally:
        await openai_async_client.close()


async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    images: Optional[List[str]] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        images=images,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    images: Optional[List[str]] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        images=images,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    images: Optional[List[str]] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        images=images,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    images: Optional[List[str]] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        base_url="https://integrate.api.nvidia.com/v1",
        images=images,
        **kwargs,
    )


