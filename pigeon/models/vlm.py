import asyncio
import openai
import base64
import os, io, time
from PIL import Image
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from datetime import datetime


_VLM_MAX_ATTEMPTS = 10
_VLM_RETRY_INTERVAL_SECONDS = 240


class VLMServiceUnavailable(BaseException):
    """Abort evaluation when retries fail, bypassing semantic-answer fallbacks."""


def _log_request_failure(error):
    print(f"{datetime.now().isoformat(timespec='seconds')} VLM request failed: {type(error).__name__}: {error}", flush=True)


def _chat_request_kwargs(model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build shared chat-completion arguments, including optional Qwen controls."""
    timeout = float(os.environ.get("VLM_TIMEOUT", "240"))
    if timeout <= 0:
        raise ValueError("VLM_TIMEOUT must be greater than 0")
    default_temperature = 0.8 if "Qwen" in model else 1
    temperature = float(os.environ.get("VLM_TEMPERATURE", default_temperature))
    if temperature < 0:
        raise ValueError("VLM_TEMPERATURE must be non-negative")
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "temperature": temperature,
    }
    raw_seed = os.environ.get("VLM_SEED")
    if raw_seed is not None and raw_seed.strip():
        request_kwargs["seed"] = int(raw_seed)
    raw_value = os.environ.get("VLM_ENABLE_THINKING")
    if raw_value is None or not raw_value.strip():
        return request_kwargs

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        enable_thinking = True
    elif normalized in {"0", "false", "no", "off"}:
        enable_thinking = False
    else:
        raise ValueError(
            "VLM_ENABLE_THINKING must be one of: true, false, 1, 0, yes, no, on, off"
        )
    request_kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }
    return request_kwargs


class VLMClient:
    def __init__(
        self,
        vlm_model,
    ):
        # Track how many VLM calls are made (per episode via reset_call_count).
        self.call_count: int = 0
        self.input_token_count: int = 0
        self.output_token_count: int = 0
        self.last_prompt: Optional[Any] = None
        self.last_images = None
        self.last_response: Optional[str] = None
        vlm_api_key = os.environ.get("VLM_API_KEY")
        vlm_base_url = os.environ.get("VLM_BASE_URL")
        if vlm_base_url:
            # OpenAI-compatible endpoint (e.g., vLLM) with optional auth.
            self.client = openai.OpenAI(
                api_key=vlm_api_key or "",
                base_url=vlm_base_url,
                max_retries=0,
            )
            self.vlm_model = vlm_model
            return
        if "gpt" not in vlm_model:
            raise ValueError("VLM_API_KEY and VLM_BASE_URL is required for non-OpenAI models.")
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            openai_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
            if openai_key is None:
                raise ValueError("No OpenAI key provided")
            self.client = openai.AzureOpenAI(max_retries=0)
        else:
            base_url = os.environ.get("OPENAI_API_BASE", None)
            self.client = openai.OpenAI(api_key=openai_key, base_url=base_url, max_retries=0)
        self.vlm_model = vlm_model


    def reset_call_count(self) -> None:
        """Reset the per-episode VLM call counter and token usage."""
        self.call_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def _record_call(self) -> None:
        """Increment the logical VLM call counter once per high-level request."""
        self.call_count += 1

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            self.input_token_count += int(prompt_tokens or 0)
            self.output_token_count += int(completion_tokens or 0)
        except (TypeError, ValueError):
            pass

    def _get_response_text(self, text, failed_response=""):
        """Text-only chat completion helper.

        Accepts either a plain string prompt or a full list of messages
        (e.g., ValuemapFusionPlanner.candidate_prompt_template style), and
        tries up to 10 times until a non-empty message is received, waiting
        240 seconds between failed attempts.
        """
        # Count this as a single logical VLM call (retries are internal).
        self._record_call()
        self.last_prompt = text
        self.last_images = None
        # Build messages according to provided format
        if isinstance(text, list):
            # Expecting a list of {"role": ..., "content": ...} messages
            messages = text
        else:
            # Fallback to a single user message with plain string content
            messages = [{"role": "user", "content": str(text)}]

        response_text = failed_response
        last_error = None
        for attempt in range(_VLM_MAX_ATTEMPTS):
            try:
                resp = self.client.chat.completions.create(
                    **_chat_request_kwargs(self.vlm_model, messages)
                )
                self._record_usage(resp)
                response_text = getattr(resp.choices[0].message, "content", "")
                if isinstance(response_text, str) and response_text.strip():
                    break
                last_error = ValueError("VLM returned an empty answer")
            except Exception as e:
                _log_request_failure(e)
                last_error = e
                response_text = failed_response
            if attempt + 1 < _VLM_MAX_ATTEMPTS:
                time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
        else:
            raise VLMServiceUnavailable(
                f"Text request failed after {_VLM_MAX_ATTEMPTS} attempts; episode must be rerun"
            ) from last_error

        self.last_response = response_text
        return response_text

    def _get_response(self, rgb, text, failed_response=""):
        # Count this as a single logical VLM call (retries are internal).
        self._record_call()
        self.last_prompt = text
        if isinstance(rgb, Tuple):
            images = rgb
        else:
            images = [rgb]
        self.last_images = images
        content = [{"type": "text", "text": text}]
        for rgb in images:
            image_pil = Image.fromarray(rgb)
            buffered = io.BytesIO()
            image_pil.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{img_base64}"
            content.append({"type": "image_url", "image_url": {"url": image_data_uri}})

        last_error = None
        for attempt in range(_VLM_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    **_chat_request_kwargs(
                        self.vlm_model,
                        [{"role": "user", "content": content}],
                    )
                )
                self._record_usage(response)
                response = response.choices[0].message.content
                if not isinstance(response, str) or not response.strip():
                    raise ValueError("VLM returned an empty answer")
            except Exception as e:
                _log_request_failure(e)
                last_error = e
                response = failed_response
                if attempt + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
            else:
                break
        else:
            raise VLMServiceUnavailable(
                f"Image request failed after {_VLM_MAX_ATTEMPTS} attempts; episode must be rerun"
            ) from last_error

        self.last_response = response
        return response


class AsyncVLMClient:
    def __init__(
        self,
        vlm_model,
    ):
        # Track how many VLM calls are made (per episode via reset_call_count).
        self.call_count: int = 0
        self.input_token_count: int = 0
        self.output_token_count: int = 0
        self.last_prompt: Optional[Any] = None
        self.last_images = None
        self.last_response: Optional[str] = None
        vlm_api_key = os.environ.get("VLM_API_KEY")
        vlm_base_url = os.environ.get("VLM_BASE_URL")
        if vlm_base_url:
            # OpenAI-compatible endpoint (e.g., vLLM) with optional auth.
            self.async_client = openai.AsyncOpenAI(
                api_key=vlm_api_key or "",
                base_url=vlm_base_url,
            )
            self.client = openai.OpenAI(
                api_key=vlm_api_key or "",
                base_url=vlm_base_url,
            )
            self.vlm_model = vlm_model
            return
        if "gpt" not in vlm_model:
            raise ValueError("VLM_API_KEY and VLM_BASE_URL is required for non-OpenAI models.")
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            openai_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
            if openai_key is None:
                raise ValueError("No OpenAI key provided")
            self.async_client = openai.AsyncAzureOpenAI()
            self.client = openai.AzureOpenAI()
        else:
            base_url = os.environ.get("OPENAI_API_BASE", None)
            self.async_client = openai.AsyncOpenAI(api_key=openai_key, base_url=base_url)
            self.client = openai.OpenAI(api_key=openai_key, base_url=base_url)
        self.vlm_model = vlm_model

    def reset_call_count(self) -> None:
        """Reset the per-episode VLM call counter and token usage."""
        self.call_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def _record_call(self) -> None:
        """Increment the logical VLM call counter once per high-level request."""
        self.call_count += 1

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            self.input_token_count += int(prompt_tokens or 0)
            self.output_token_count += int(completion_tokens or 0)
        except (TypeError, ValueError):
            pass

    def _get_response_text(self, text, failed_response=""):
        """Text-only chat completion helper.

        Accepts either a plain string prompt or a full list of messages
        (e.g., ValuemapFusionPlanner.candidate_prompt_template style), and
        retries up to 10 times until a non-empty message is received.
        """
        # Count this as a single logical VLM call (retries are internal).
        self.call_count += 1
        self.last_prompt = text
        self.last_images = None
        # Build messages according to provided format
        if isinstance(text, list):
            # Expecting a list of {"role": ..., "content": ...} messages
            messages = text
        else:
            # Fallback to a single user message with plain string content
            messages = [{"role": "user", "content": str(text)}]

        response_text = failed_response
        for attempt in range(_VLM_MAX_ATTEMPTS):  # retry to ensure connection and non-empty reply
            try:
                resp = self.client.chat.completions.create(
                    **_chat_request_kwargs(self.vlm_model, messages)
                )
                self._record_usage(resp)
                response_text = getattr(resp.choices[0].message, "content", "")
                if isinstance(response_text, str) and response_text.strip():
                    break
                if attempt + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
            except TimeoutError as e:
                print("Timeout error in OpenAI API call:", e)
                response_text = failed_response
                if attempt + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
            except Exception as e:
                print("Error in OpenAI API call:", e)
                response_text = failed_response
                if attempt + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)

        self.last_response = response_text
        return response_text

    def _get_response(self, rgb, text, failed_response=""):
        # Count this as a single logical VLM call (retries are internal).
        self._record_call()
        self.last_prompt = text
        if isinstance(rgb, Tuple):
            images = rgb
        else:
            images = [rgb]
        content = [{"type": "text", "text": text}]
        for rgb in images:
            image_pil = Image.fromarray(rgb)
            buffered = io.BytesIO()
            image_pil.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{img_base64}"
            content.append({"type": "image_url", "image_url": {"url": image_data_uri}})

        for i in range(_VLM_MAX_ATTEMPTS):  # to ensure net connection
            success = True
            try:
                response = self.client.chat.completions.create(
                    **_chat_request_kwargs(
                        self.vlm_model,
                        [{"role": "user", "content": content}],
                    )
                )
                self._record_usage(response)
                response = response.choices[0].message.content
            except TimeoutError as e:
                print("Timeout error in OpenAI API call:", e)
                response = failed_response
                success = False
                if i + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
            except Exception as e:
                print("Error in OpenAI API call:", e)
                response = failed_response
                success = False
                if i + 1 < _VLM_MAX_ATTEMPTS:
                    time.sleep(_VLM_RETRY_INTERVAL_SECONDS)
            if success:
                break

        self.last_response = response
        return response

    async def _async_get_response(self, rgb, text, failed_response=""):
        # Count this as a single logical VLM call (retries are internal).
        self.call_count += 1
        self.last_prompt = text
        if isinstance(rgb, (Tuple, List)):
            images = rgb
        else:
            images = [rgb]
        content = [{"type": "text", "text": text}]
        for rgb in images:
            image_pil = Image.fromarray(rgb)
            buffered = io.BytesIO()
            image_pil.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{img_base64}"
            content.append({"type": "image_url", "image_url": {"url": image_data_uri}})

        for i in range(_VLM_MAX_ATTEMPTS):  # to ensure net connection
            success = True
            try:
                response = await self.async_client.chat.completions.create(
                    **_chat_request_kwargs(
                        self.vlm_model,
                        [{"role": "user", "content": content}],
                    )
                )
                self._record_usage(response)
                response = response.choices[0].message.content
                if not isinstance(response, str) or not response.strip():
                    response = failed_response
                    success = False
            except TimeoutError as e:
                print("Timeout error in OpenAI API call:", e)
                response = failed_response
                success = False
            except Exception as e:
                print("Error in OpenAI API call:", e)
                response = failed_response
                success = False
            if success:
                break
            if i + 1 < _VLM_MAX_ATTEMPTS:
                await asyncio.sleep(_VLM_RETRY_INTERVAL_SECONDS)

        return response
