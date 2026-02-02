import asyncio
from typing import List, Optional, Union, AsyncIterator, Any
import os
from functools import partial

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from lightrag.utils import (
    logger,
)
from lightrag.api import __api_version__

from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

import ollama




@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)

def _load_images_as_base64(image_paths: Optional[List[str]]) -> List[str]:
    """
    Đọc list path ảnh và trả về list base64 string theo format Ollama yêu cầu.
    Nếu path lỗi / không đọc được thì skip và log cảnh báo.
    """
    if not image_paths:
        return []

    import base64

    encoded_images: List[str] = []
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                encoded_images.append(encoded)
        except Exception as e:
            logger.warning(f"Failed to load image '{p}': {e}")
            continue
    return encoded_images

async def _ollama_model_if_cache(
    model,
    prompt,
    images: Optional[List[str]] = None,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """
    VLM wrapper cho Ollama: giống _ollama_model_if_cache nhưng hỗ trợ images (list path).

    - Nếu có images: encode base64 và gắn vào message cuối cùng của user.
    - Nếu không có images: chạy như LLM text bình thường.
    """
    if enable_cot:
        logger.debug(
            "enable_cot=True is not specially handled for ollama VLM and will be ignored."
        )

    stream = True if kwargs.get("stream") else False

    # Dọn kwargs giống bản text
    kwargs.pop("max_tokens", None)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # history_messages đã là list dict kiểu {"role": "...", "content": "..."}
        messages.extend(history_messages)

        # User message cuối cùng (nơi ta gắn images nếu có)
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}

        encoded_images = _load_images_as_base64(images)
        if encoded_images:
            # Ollama VLM nhận images ngay trong message của user
            user_msg["images"] = encoded_images

        messages.append(user_msg)

        # Gọi Ollama chat
        response = await ollama_client.chat(model=model, messages=messages, **kwargs)

        if stream:
            # không cache / xử lý reasoning cho stream như bản text
            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in VLM stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug(
                            "Successfully closed Ollama client for VLM streaming"
                        )
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close Ollama client (VLM stream): "
                            f"{close_error}"
                        )

            return inner()
        else:
            model_response = response["message"]["content"]
            return model_response

    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama VLM client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama VLM client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama VLM client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama VLM client in finally block: {close_error}"
                )


async def ollama_model_complete(
    prompt,
    images: Optional[List[str]] = None,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        images=images,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )