"""
Fan-out coordinator that dispatches a single prompt to every persona
in parallel through asyncio.gather and returns the responses in the
original persona order.

Any unexpected exception bubbling out of the underlying client (which
should normally return a ModelFailure rather than raise) is captured
by gather(return_exceptions=True) and converted into a synthetic
ModelFailure so the caller never has to handle raw exceptions.

---

Tek bir prompt'u, asyncio.gather aracılığıyla her personaya paralel
olarak dağıtan ve cevapları orijinal persona sırasında döndüren
fan-out koordinatörü.

Alttaki client'tan beklenmedik şekilde sızan bir exception (normalde
exception yerine ModelFailure dönmesi beklenir), gather'ın
return_exceptions=True modu tarafından yakalanır ve sentetik bir
ModelFailure'a dönüştürülür; böylece çağıran taraf ham exception ile
uğraşmaz.
"""

import asyncio
from typing import Final

from core.gemini_client import GeminiClient
from core.models import ModelFailure, ModelResponse, PromptRequest

# Error labels for the synthetic-failure path / Sentetik hata yolu için etiketler
_ORCHESTRATOR_ERROR_TYPE: Final = "OrchestratorError"
_UNKNOWN_MODEL_NAME: Final = "unknown"


class Orchestrator:
    """
    Fan-out coordinator over a single GeminiClient.

    The same client is reused across all persona calls so the
    underlying HTTP/2 connection pool can be shared; concurrency comes
    from asyncio.gather, not from spawning multiple clients.

    ---

    Tek bir GeminiClient üzerinde çalışan fan-out koordinatörü.

    Tüm persona çağrılarında aynı client kullanılır; böylece alttaki
    HTTP/2 bağlantı havuzu paylaşılabilir. Eşzamanlılık birden çok
    client oluşturmaktan değil, asyncio.gather'dan gelir.
    """

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def run(self, request: PromptRequest) -> list[ModelResponse]:
        """
        Fan out the request to all personas and gather responses in
        the original persona order.

        ---

        İsteği tüm personalara fan-out yapar ve cevapları orijinal
        persona sırasında toplar.
        """
        coros = [
            self._client.call(request.prompt, persona) for persona in request.personas
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        results: list[ModelResponse] = []
        for persona, result in zip(request.personas, raw_results, strict=True):
            if isinstance(result, BaseException):
                results.append(
                    ModelFailure(
                        persona_name=persona.name,
                        model_name=_UNKNOWN_MODEL_NAME,
                        latency_ms=0.0,
                        error_type=_ORCHESTRATOR_ERROR_TYPE,
                        error_message=f"{type(result).__name__}: {result}",
                    )
                )
            else:
                results.append(result)

        return results
