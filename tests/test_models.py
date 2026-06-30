"""
Unit tests for core.models — Persona, PromptRequest, ModelResponse
union, and the default-persona constants.

---

core.models birim testleri — Persona, PromptRequest, ModelResponse
union'ı ve varsayılan persona sabitleri.
"""

from dataclasses import FrozenInstanceError

import pytest

from core.models import (
    CREATIVE,
    DEFAULT_PERSONAS,
    ModelFailure,
    ModelResponse,
    ModelSuccess,
    Persona,
    PromptRequest,
    response_to_dict,
)


class TestPersona:
    def test_valid_construction(self) -> None:
        p = Persona(name="x", temperature=0.5, system_prompt="hi")
        assert p.name == "x"
        assert p.temperature == 0.5
        assert p.system_prompt == "hi"

    def test_system_prompt_defaults_to_none(self) -> None:
        p = Persona(name="x", temperature=0.5)
        assert p.system_prompt is None

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name cannot be empty"):
            Persona(name="", temperature=0.5)

    @pytest.mark.parametrize("temp", [-0.1, 2.1, -1.0, 5.0])
    def test_temperature_out_of_range(self, temp: float) -> None:
        with pytest.raises(ValueError, match="outside"):
            Persona(name="x", temperature=temp)

    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0, 2.0])
    def test_temperature_boundaries_accepted(self, temp: float) -> None:
        # Inclusive bounds at 0.0 and 2.0 are valid.
        # 0.0 ve 2.0 dahil olmak üzere sınırlar geçerli.
        Persona(name="x", temperature=temp)

    def test_frozen_instance(self) -> None:
        p = Persona(name="x", temperature=0.5)
        with pytest.raises(FrozenInstanceError):
            p.name = "y"  # type: ignore[misc]


class TestPromptRequest:
    def test_valid_construction(self) -> None:
        req = PromptRequest(prompt="hi", personas=(CREATIVE,))
        assert req.prompt == "hi"
        assert req.personas == (CREATIVE,)

    def test_empty_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            PromptRequest(prompt="", personas=(CREATIVE,))

    def test_whitespace_only_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            PromptRequest(prompt="   \t\n", personas=(CREATIVE,))

    def test_empty_personas_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PromptRequest(prompt="hi", personas=())

    def test_duplicate_persona_names_raise(self) -> None:
        twin = Persona(name="creative", temperature=0.1)
        with pytest.raises(ValueError, match="Duplicate"):
            PromptRequest(prompt="hi", personas=(CREATIVE, twin))


class TestDefaults:
    def test_three_personas(self) -> None:
        assert len(DEFAULT_PERSONAS) == 3

    def test_ordered_names(self) -> None:
        assert [p.name for p in DEFAULT_PERSONAS] == [
            "creative",
            "balanced",
            "precise",
        ]

    def test_temperatures_match_spec(self) -> None:
        assert [p.temperature for p in DEFAULT_PERSONAS] == [0.9, 0.5, 0.1]

    def test_balanced_has_no_system_prompt(self) -> None:
        balanced = DEFAULT_PERSONAS[1]
        assert balanced.system_prompt is None


class TestModelResponse:
    def test_success_construction(self) -> None:
        r = ModelSuccess(
            persona_name="x",
            model_name="m",
            latency_ms=100.0,
            text="hi",
            tokens=10,
        )
        assert r.text == "hi"
        assert r.tokens == 10

    def test_failure_construction(self) -> None:
        r = ModelFailure(
            persona_name="x",
            model_name="m",
            latency_ms=100.0,
            error_type="X",
            error_message="oops",
        )
        assert r.error_type == "X"

    def test_discriminated_union_narrowing(self) -> None:
        # The union should narrow correctly via isinstance.
        # Union, isinstance ile doğru şekilde daralmalı.
        def handle(r: ModelResponse) -> str:
            if isinstance(r, ModelSuccess):
                return r.text
            return r.error_type

        s: ModelResponse = ModelSuccess(
            persona_name="x",
            model_name="m",
            latency_ms=0.0,
            text="ok",
            tokens=1,
        )
        f: ModelResponse = ModelFailure(
            persona_name="x",
            model_name="m",
            latency_ms=0.0,
            error_type="RateLimit",
            error_message="m",
        )
        assert handle(s) == "ok"
        assert handle(f) == "RateLimit"


class TestResponseToDict:
    def test_success_carries_kind_and_fields(self) -> None:
        r = ModelSuccess(
            persona_name="creative",
            model_name="gemini-2.5-flash",
            latency_ms=1234.5,
            text="Paris",
            tokens=42,
        )
        d = response_to_dict(r)
        assert d == {
            "kind": "success",
            "persona_name": "creative",
            "model_name": "gemini-2.5-flash",
            "latency_ms": 1234.5,
            "text": "Paris",
            "tokens": 42,
        }

    def test_failure_carries_kind_and_fields(self) -> None:
        r = ModelFailure(
            persona_name="precise",
            model_name="gemini-2.5-flash",
            latency_ms=512.0,
            error_type="RateLimit",
            error_message="throttled",
        )
        d = response_to_dict(r)
        assert d == {
            "kind": "failure",
            "persona_name": "precise",
            "model_name": "gemini-2.5-flash",
            "latency_ms": 512.0,
            "error_type": "RateLimit",
            "error_message": "throttled",
        }

    def test_discriminator_distinguishes_shapes(self) -> None:
        # A JSON consumer should branch on "kind" alone.
        # JSON tüketicisi yalnızca "kind" üzerinden dallanabilmeli.
        s = response_to_dict(
            ModelSuccess(
                persona_name="x",
                model_name="m",
                latency_ms=0.0,
                text="ok",
                tokens=1,
            )
        )
        f = response_to_dict(
            ModelFailure(
                persona_name="x",
                model_name="m",
                latency_ms=0.0,
                error_type="X",
                error_message="m",
            )
        )
        assert s["kind"] == "success"
        assert f["kind"] == "failure"
        assert "text" in s and "text" not in f
        assert "error_type" in f and "error_type" not in s

    def test_dict_is_json_serializable(self) -> None:
        import json

        r = ModelSuccess(
            persona_name="x",
            model_name="m",
            latency_ms=1.5,
            text="merhaba 🌍",
            tokens=3,
        )
        encoded = json.dumps(response_to_dict(r), ensure_ascii=False)
        assert "merhaba 🌍" in encoded
