"""
Async wrapper around the google-genai SDK with retry, jittered
exponential backoff, and per-call timeout. Every call returns a
ModelResponse — either a ModelSuccess (text + token count) or a
ModelFailure (with an error_type label) — so the caller never has
to handle raw SDK exceptions.

---

google-genai SDK'sının async sarmalayıcısı; retry, jitter'lı üstel
backoff ve çağrı başı timeout içerir. Her çağrı ModelResponse döner —
ya ModelSuccess (metin + token sayısı) ya da ModelFailure (error_type
etiketi ile) — böylece çağıran tarafın ham SDK exception'larını ele
alması gerekmez.
"""

import asyncio
import os
import random
import time
from typing import Final

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core.models import ModelFailure, ModelResponse, ModelSuccess, Persona

# Configuration defaults / Yapılandırma varsayılanları
_DEFAULT_MODEL_NAME: Final = "gemini-2.5-flash"
_DEFAULT_MAX_RETRIES: Final = 3
_DEFAULT_INITIAL_BACKOFF_S: Final = 1.0
_DEFAULT_TIMEOUT_S: Final = 30.0
_JITTER_MAX_S: Final = 0.5


class GeminiClient:
    """
    Async client for a single Gemini model with retry and rate-limit
    handling. Construction reads GEMINI_API_KEY from the environment
    unless an explicit api_key is passed; each instance owns its own
    underlying genai.Client so multiple clients do not share state.

    ---

    Tek bir Gemini modeli için async client; retry ve rate-limit
    yönetimi içerir. İnşa sırasında, açık bir api_key verilmediyse,
    GEMINI_API_KEY ortam değişkeninden okunur; her örnek kendi alttaki
    genai.Client'ına sahiptir, birden çok client durum paylaşmaz.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_backoff_s: float = _DEFAULT_INITIAL_BACKOFF_S,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not set; pass api_key= or export the env var"
            )
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if initial_backoff_s <= 0:
            raise ValueError("initial_backoff_s must be > 0")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model_name
        self._max_retries = max_retries
        self._initial_backoff_s = initial_backoff_s
        self._timeout_s = timeout_s

    async def call(self, prompt: str, persona: Persona) -> ModelResponse:
        """
        Run a single (prompt, persona) call with retry on transient
        errors. Returns ModelSuccess on success, ModelFailure on any
        unrecoverable or out-of-retries error — never raises for
        upstream API failures.

        ---

        Tek bir (prompt, persona) çağrısını geçici hatalara karşı
        retry ile çalıştırır. Başarıda ModelSuccess, kurtarılamayan
        veya retry'ları tükenmiş hatada ModelFailure döner — upstream
        API hataları için exception fırlatmaz.
        """
        for attempt in range(self._max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                return await self._call_once(prompt, persona)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - attempt_started) * 1000
                error_type, retryable = self._classify_error(exc)
                if not retryable or attempt == self._max_retries:
                    return ModelFailure(
                        persona_name=persona.name,
                        model_name=self._model_name,
                        latency_ms=elapsed_ms,
                        error_type=error_type,
                        error_message=str(exc),
                    )
                backoff = self._initial_backoff_s * (2**attempt) + random.uniform(
                    0, _JITTER_MAX_S
                )
                await asyncio.sleep(backoff)

        # Loop guarantees a return inside; this is unreachable.
        # Döngü içinde return garantili; bu satıra ulaşılamaz.
        raise RuntimeError("GeminiClient.call: retry loop exited without return")

    async def _call_once(self, prompt: str, persona: Persona) -> ModelSuccess:
        config = genai_types.GenerateContentConfig(
            temperature=persona.temperature,
            system_instruction=persona.system_prompt,
        )

        started = time.perf_counter()
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            ),
            timeout=self._timeout_s,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        tokens = (
            int(response.usage_metadata.total_token_count)
            if response.usage_metadata is not None
            and response.usage_metadata.total_token_count is not None
            else 0
        )

        return ModelSuccess(
            persona_name=persona.name,
            model_name=self._model_name,
            latency_ms=elapsed_ms,
            text=str(response.text),
            tokens=tokens,
        )

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[str, bool]:
        """
        Map an SDK exception to (error_type label, retryable flag).
        Classification is by HTTP status code: 429 + 5xx are
        retryable, 4xx and unknown are not.

        ---

        Bir SDK exception'ını (error_type etiketi, retry-edilebilir
        bayrağı) çiftine eşler. Sınıflandırma HTTP status koduna göre:
        429 ve 5xx retry edilir, 4xx ve bilinmeyen edilmez.
        """
        if isinstance(exc, TimeoutError):
            return ("Timeout", True)

        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429:
                return ("RateLimit", True)
            if isinstance(code, int) and 500 <= code < 600:
                return ("ServerError", True)
            if code == 400:
                return ("InvalidArgument", False)
            if code in (401, 403):
                return ("AuthError", False)
            if code == 404:
                return ("NotFound", False)
            if isinstance(code, int) and 400 <= code < 500:
                return ("ClientError", False)
            return ("APIError", False)

        if isinstance(exc, ValueError):
            # response.text raises ValueError on blocked/empty content.
            # response.text engellenmiş/boş içerikte ValueError fırlatır.
            return ("EmptyResponse", False)

        return ("UnknownError", False)
