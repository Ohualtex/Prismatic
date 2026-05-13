"""
Unit tests for core.orchestrator — fan-out, order preservation, and
the synthetic-failure path for unexpected exceptions from the client.

---

core.orchestrator birim testleri — fan-out, sıra koruması ve
client'tan gelen beklenmedik exception'lar için sentetik hata yolu.
"""

import asyncio
from unittest.mock import AsyncMock

from core.gemini_client import GeminiClient
from core.models import (
    CREATIVE,
    DEFAULT_PERSONAS,
    ModelFailure,
    ModelSuccess,
    Persona,
    PromptRequest,
)
from core.orchestrator import Orchestrator


def _success(name: str, text: str = "ok") -> ModelSuccess:
    return ModelSuccess(
        persona_name=name,
        model_name="mock",
        latency_ms=10.0,
        text=text,
        tokens=1,
    )


def _failure(name: str, error_type: str = "X") -> ModelFailure:
    return ModelFailure(
        persona_name=name,
        model_name="mock",
        latency_ms=10.0,
        error_type=error_type,
        error_message="oops",
    )


async def test_all_success() -> None:
    client = AsyncMock(spec=GeminiClient)
    client.call.side_effect = [
        _success("creative", "C"),
        _success("balanced", "B"),
        _success("precise", "P"),
    ]
    orch = Orchestrator(client)

    results = await orch.run(PromptRequest(prompt="hi", personas=DEFAULT_PERSONAS))

    assert len(results) == 3
    assert all(isinstance(r, ModelSuccess) for r in results)
    assert [r.persona_name for r in results] == ["creative", "balanced", "precise"]


async def test_mixed_success_and_failure() -> None:
    client = AsyncMock(spec=GeminiClient)
    client.call.side_effect = [
        _success("creative"),
        _failure("balanced", "RateLimit"),
        _success("precise"),
    ]
    orch = Orchestrator(client)

    results = await orch.run(PromptRequest(prompt="hi", personas=DEFAULT_PERSONAS))

    assert isinstance(results[0], ModelSuccess)
    assert isinstance(results[1], ModelFailure)
    assert results[1].error_type == "RateLimit"
    assert isinstance(results[2], ModelSuccess)


async def test_order_preserved_under_async_jitter() -> None:
    # Out-of-order completion must not change the result order.
    # Sırasız tamamlanma sonuç sırasını değiştirmemeli.
    async def staggered_call(_prompt: str, persona: Persona) -> ModelSuccess:
        delays = {"creative": 0.03, "balanced": 0.01, "precise": 0.02}
        await asyncio.sleep(delays[persona.name])
        return _success(persona.name)

    client = AsyncMock(spec=GeminiClient)
    client.call = AsyncMock(side_effect=staggered_call)
    orch = Orchestrator(client)

    results = await orch.run(PromptRequest(prompt="hi", personas=DEFAULT_PERSONAS))

    assert [r.persona_name for r in results] == ["creative", "balanced", "precise"]


async def test_unexpected_exception_wrapped_as_failure() -> None:
    client = AsyncMock(spec=GeminiClient)
    client.call.side_effect = [
        _success("creative"),
        RuntimeError("simulated client bug"),
        _success("precise"),
    ]
    orch = Orchestrator(client)

    results = await orch.run(PromptRequest(prompt="hi", personas=DEFAULT_PERSONAS))

    assert isinstance(results[0], ModelSuccess)
    assert isinstance(results[1], ModelFailure)
    assert results[1].error_type == "OrchestratorError"
    assert "RuntimeError" in results[1].error_message
    assert "simulated client bug" in results[1].error_message
    assert isinstance(results[2], ModelSuccess)


async def test_single_persona() -> None:
    client = AsyncMock(spec=GeminiClient)
    client.call = AsyncMock(return_value=_success("creative"))
    orch = Orchestrator(client)

    results = await orch.run(PromptRequest(prompt="hi", personas=(CREATIVE,)))

    assert len(results) == 1
    assert results[0].persona_name == "creative"
