"""
Unit tests for main.py helpers — the API-key setup check that
pre-validates GEMINI_API_KEY before any network call, plus the
--json flag parsing and JSON output rendering.

---

main.py yardımcılarının birim testleri — herhangi bir ağ çağrısından
önce GEMINI_API_KEY'i doğrulayan kurulum kontrolü, ayrıca --json
bayrağı ayrıştırması ve JSON çıktı render'ı.
"""

import json

import pytest

import main
from core.models import ModelFailure, ModelSuccess


class TestApiKeySetupError:
    def test_missing_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        msg = main._api_key_setup_error()
        assert msg is not None
        assert "not set" in msg

    def test_empty_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "")
        msg = main._api_key_setup_error()
        assert msg is not None
        assert "not set" in msg

    def test_whitespace_only_returns_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "   \t\n")
        msg = main._api_key_setup_error()
        assert msg is not None
        assert "not set" in msg

    def test_placeholder_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "paste_your_real_key_here")
        msg = main._api_key_setup_error()
        assert msg is not None
        assert "placeholder" in msg

    def test_real_looking_key_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyA-real-looking-fake")
        assert main._api_key_setup_error() is None


class TestJsonFlag:
    def test_json_flag_defaults_false(self) -> None:
        args = main._build_argparser().parse_args(["hello"])
        assert args.json is False

    def test_json_flag_sets_true(self) -> None:
        args = main._build_argparser().parse_args(["hello", "--json"])
        assert args.json is True


class TestPrintJson:
    def test_outputs_valid_json_array(self, capsys: pytest.CaptureFixture[str]) -> None:
        responses = [
            ModelSuccess(
                persona_name="creative",
                model_name="m",
                latency_ms=1.0,
                text="Paris",
                tokens=5,
            ),
            ModelFailure(
                persona_name="balanced",
                model_name="m",
                latency_ms=2.0,
                error_type="RateLimit",
                error_message="throttled",
            ),
        ]
        main._print_json(responses)

        parsed = json.loads(capsys.readouterr().out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["kind"] == "success"
        assert parsed[0]["text"] == "Paris"
        assert parsed[1]["kind"] == "failure"
        assert parsed[1]["error_type"] == "RateLimit"

    def test_unicode_preserved(self, capsys: pytest.CaptureFixture[str]) -> None:
        responses = [
            ModelSuccess(
                persona_name="x",
                model_name="m",
                latency_ms=1.0,
                text="merhaba 🌍",
                tokens=3,
            ),
        ]
        main._print_json(responses)
        # Raw output keeps the literal characters, not \uXXXX escapes.
        # Ham çıktı \uXXXX kaçışları yerine gerçek karakterleri korur.
        assert "merhaba 🌍" in capsys.readouterr().out
