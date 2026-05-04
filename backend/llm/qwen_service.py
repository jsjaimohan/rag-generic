"""LM Studio OpenAI-compatible chat completions (Phi, Llama, Qwen, etc.)."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import httpx

from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_TRANSIENT_HTTP_STATUS = frozenset({429, 502, 503, 504})

T = TypeVar("T")


def _with_lmstudio_retries(
    operation: str,
    fn: Callable[[], T],
    *,
    retry_read_timeout: bool,
) -> T:
    """Run ``fn`` with retries on connection failures and transient HTTP statuses."""
    attempts = settings.lmstudio_http_attempts
    backoff = max(0.0, settings.lmstudio_retry_backoff_seconds)
    for attempt in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _TRANSIENT_HTTP_STATUS or attempt >= attempts - 1:
                raise
            err: BaseException = exc
        except httpx.RequestError as exc:
            if isinstance(exc, httpx.ReadTimeout) and not retry_read_timeout:
                raise
            if attempt >= attempts - 1:
                raise
            err = exc
        delay = backoff * (2**attempt)
        logger.warning(
            "lmstudio.retry operation=%s attempt=%s/%s err=%s sleep_s=%.2f",
            operation,
            attempt + 1,
            attempts,
            err,
            delay,
        )
        time.sleep(delay)
    raise RuntimeError("lmstudio retries exhausted")  # pragma: no cover


class QwenService:
    """LM Studio ``/v1/chat/completions`` client (model id from ``LMSTUDIO_MODEL``)."""

    def health_check(self) -> dict:
        """Validate LM Studio endpoint and model listing endpoint."""
        headers = {"Authorization": f"Bearer {settings.lmstudio_api_key}"}

        def _call() -> dict:
            response = httpx.get(
                f"{settings.lmstudio_base_url}/models",
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
            model_ids = [item.get("id", "") for item in payload.get("data", [])]
            return {
                "ok": True,
                "models_count": len(model_ids),
                "configured_model_found": settings.lmstudio_model in model_ids,
            }

        return _with_lmstudio_retries("health_models", _call, retry_read_timeout=True)

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        timeout_seconds: int | None = None,
    ) -> str:
        """Raw chat completion (multi-message) for structured prompts."""
        read_timeout = (
            settings.lmstudio_chat_read_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        # LMSTUDIO_CHAT_READ_TIMEOUT_SECONDS=0 disables the read cap (use a high ceiling).
        if read_timeout <= 0:
            read_timeout = 86400
        headers = {
            "Authorization": f"Bearer {settings.lmstudio_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": settings.lmstudio_model,
            "messages": messages,
            "temperature": temperature,
        }
        # Qwen3 “thinking” mode — unsupported on Phi/Llama/Gemma; leave LMSTUDIO_CHAT_DISABLE_THINKING=false.
        if settings.lmstudio_chat_disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        # Split timeouts so a dead LM Studio host fails on connect quickly (not multi‑minute hangs).
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(read_timeout),
            write=30.0,
            pool=5.0,
        )

        def _parse_completion_response(response: httpx.Response) -> str:
            response_payload = response.json()
            choices = response_payload.get("choices", [])
            if not choices:
                raise ValueError("LLM returned no choices.")
            message = choices[0].get("message", {})
            return str(message.get("content", "")).strip()

        def _call() -> str:
            response = httpx.post(
                f"{settings.lmstudio_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            # Many LM Studio builds reject Qwen-only fields (e.g. chat_template_kwargs) with 400.
            if response.status_code == 400 and "chat_template_kwargs" in payload:
                body_preview = (response.text or "")[:2000]
                logger.warning(
                    "lmstudio.chat_completions_400_retry_without_template_kwargs body_preview=%s",
                    body_preview,
                )
                slim = {k: v for k, v in payload.items() if k != "chat_template_kwargs"}
                response = httpx.post(
                    f"{settings.lmstudio_base_url}/chat/completions",
                    headers=headers,
                    json=slim,
                    timeout=timeout,
                )
            if response.status_code >= 400:
                body = (response.text or "")[:4000]
                logger.error(
                    "lmstudio.chat_completions_http_error status=%s body=%s",
                    response.status_code,
                    body,
                )
                raise RuntimeError(
                    f"LM Studio HTTP {response.status_code} at chat/completions: {body}"
                )
            return _parse_completion_response(response)

        # Do not retry read timeouts: the server may still complete the first request.
        return _with_lmstudio_retries(
            "chat_completions", _call, retry_read_timeout=False
        )

    def generate_chat_response(self, prompt: str) -> str:
        """Generate answer from LM Studio chat completions endpoint."""
        return self.complete_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a student counsellor in this chat. Follow the user message: "
                        "only state information that is supported by the supplied context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
