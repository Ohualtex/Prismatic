"""
Unit tests for core.gemini_client — construction, parameter
validation, error classification, and retry behaviour against a
mocked underlying SDK client.

---

core.gemini_client birim testleri — inşa, parametre doğrulama, hata
sınıflandırması ve mock'lanmış alttaki SDK client'ına karşı retry
davranışı.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from core.gemini_client import GeminiClient
from core.models import PRECISE, ModelFailure, ModelSuccess


@pytest.fixture
def client() -> GeminiClient:
    # max_retries=2 → up to 3 total attempts per call.
    # max_retries=2 → çağrı başına en çok 3 deneme.
    return GeminiClient(api_key="dummy-key", max_retries=2, timeout_s=5.0)


class TestConstruction:
    def test_explicit_key(self) -> None:
        c = GeminiClient(api_key="k")
        assert c._model_name == "gemini-2.5-flash"

    def test_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        c = GeminiClient()
        assert c._client is not None

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
            GeminiClient()

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"max_retries": -1}, "max_retries"),
            ({"initial_backoff_s": 0}, "initial_backoff_s"),
            ({"initial_backoff_s": -0.5}, "initial_backoff_s"),
            ({"timeout_s": 0}, "timeout_s"),
            ({"timeout_s": -1}, "timeout_s"),
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            GeminiClient(api_key="k", **kwargs)


class TestClassifyError:
    @pytest.mark.parametrize(
        "code,expected",
        [
            (429, ("RateLimit", True)),
            (500, ("ServerError", True)),
            (502, ("ServerError", True)),
            (503, ("ServerError", True)),
            (504, ("ServerError", True)),
            (400, ("InvalidArgument", False)),
            (401, ("AuthError", False)),
            (403, ("AuthError", False)),
            (404, ("NotFound", False)),
            (418, ("ClientError", False)),
        ],
    )
    def test_api_error_codes(self, code: int, expected: tuple[str, bool]) -> None:
        exc = genai_errors.APIError(
            code=code,
            response_json={"error": {"message": "x"}},
            response=None,
        )
        assert GeminiClient._classify_error(exc) == expected

    def test_timeout(self) -> None:
        assert GeminiClient._classify_error(TimeoutError("t")) == (
            "Timeout",
            True,
        )

    def test_value_error_maps_to_empty_response(self) -> None:
        assert GeminiClient._classify_error(ValueError("blocked")) == (
            "EmptyResponse",
            False,
        )

    def test_unknown_exception(self) -> None:
        assert GeminiClient._classify_error(RuntimeError("?")) == (
            "UnknownError",
            False,
        )


def _make_success_response(text: str = "ok", tokens: int = 5) -> MagicMock:
    """Build a Mock that quacks like a GenerateContentResponse.

    ---

    GenerateContentResponse'a benzer davranan bir Mock oluştur."""
    response = MagicMock()
    response.text = text
    response.usage_metadata.total_token_count = tokens
    return response


def _rate_limit() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code=429,
        response_json={"error": {"message": "throttled"}},
        response=None,
    )


def _bad_request() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code=400,
        response_json={"error": {"message": "bad"}},
        response=None,
    )


class TestCallWithRetry:
    async def test_success_first_attempt(self, client: GeminiClient) -> None:
        mock_call = AsyncMock(return_value=_make_success_response("Paris", 42))
        client._client.aio.models.generate_content = mock_call

        result = await client.call("What is the capital?", PRECISE)

        assert isinstance(result, ModelSuccess)
        assert result.text == "Paris"
        assert result.tokens == 42
        assert mock_call.call_count == 1

    async def test_retry_then_success(self, client: GeminiClient) -> None:
        client._initial_backoff_s = 0.001  # speed up retries / retry'ları hızlandır
        mock_call = AsyncMock(
            side_effect=[_rate_limit(), _rate_limit(), _make_success_response("ok")]
        )
        client._client.aio.models.generate_content = mock_call

        result = await client.call("hi", PRECISE)

        assert isinstance(result, ModelSuccess)
        assert result.text == "ok"
        assert mock_call.call_count == 3

    async def test_exhausts_retries(self, client: GeminiClient) -> None:
        client._initial_backoff_s = 0.001
        mock_call = AsyncMock(side_effect=[_rate_limit()] * 10)
        client._client.aio.models.generate_content = mock_call

        result = await client.call("hi", PRECISE)

        assert isinstance(result, ModelFailure)
        assert result.error_type == "RateLimit"
        # max_retries=2 → 3 total attempts / 3 toplam deneme
        assert mock_call.call_count == 3

    async def test_non_retryable_fails_immediately(self, client: GeminiClient) -> None:
        mock_call = AsyncMock(side_effect=_bad_request())
        client._client.aio.models.generate_content = mock_call

        result = await client.call("hi", PRECISE)

        assert isinstance(result, ModelFailure)
        assert result.error_type == "InvalidArgument"
        # 4xx → no retry / 4xx → retry yok
        assert mock_call.call_count == 1
