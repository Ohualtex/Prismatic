"""
Unit tests for main.py helpers — currently the API-key setup check
that pre-validates GEMINI_API_KEY before any network call.

---

main.py yardımcılarının birim testleri — şu an için herhangi bir ağ
çağrısından önce GEMINI_API_KEY'i doğrulayan kurulum kontrolü.
"""

import pytest

import main


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
