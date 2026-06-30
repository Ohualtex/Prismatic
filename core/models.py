"""
Data models for Prismatic — Persona, PromptRequest, ModelSuccess,
ModelFailure, and the ModelResponse discriminated union.

Frozen dataclasses are used so model instances are immutable value
objects; the discriminated union ModelResponse = ModelSuccess | ModelFailure
forces callers to handle both outcomes explicitly when type-checked
with mypy --strict.

---

Prismatic'in veri modelleri — Persona, PromptRequest, ModelSuccess,
ModelFailure ve ModelResponse ayrımlı union'ı.

Frozen dataclass'lar kullanılır, böylece model örnekleri değişmez değer
nesnelerine dönüşür; ayrımlı union ModelResponse = ModelSuccess | ModelFailure
yapısı, mypy --strict ile tip denetiminde çağıranı her iki olası sonucu
açıkça ele almaya zorlar.
"""

from dataclasses import asdict, dataclass
from typing import Final

# Gemini API supports temperature in [0.0, 2.0]; default is 1.0.
# Gemini API sıcaklık aralığı [0.0, 2.0]; default değer 1.0.
_TEMPERATURE_MIN: Final = 0.0
_TEMPERATURE_MAX: Final = 2.0


@dataclass(frozen=True, slots=True)
class Persona:
    """
    Named sampling configuration applied to a single model call.

    The temperature controls output randomness; the optional
    system_prompt sets persona tone without changing the user prompt.

    ---

    Tek bir model çağrısına uygulanan, isimlendirilmiş örnekleme
    yapılandırması. Sıcaklık çıktının rastgeleliğini kontrol eder;
    opsiyonel system_prompt kullanıcı promptunu değiştirmeden persona
    tonunu belirler.
    """

    name: str
    temperature: float
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Persona.name cannot be empty")
        if not _TEMPERATURE_MIN <= self.temperature <= _TEMPERATURE_MAX:
            raise ValueError(
                f"Persona.temperature {self.temperature} outside "
                f"[{_TEMPERATURE_MIN}, {_TEMPERATURE_MAX}]"
            )


@dataclass(frozen=True, slots=True)
class PromptRequest:
    """
    A user prompt plus the personas it should be fanned out to.

    Personas are stored as a tuple to preserve order (display order in
    the UI) and to stay consistent with the frozen=True semantics.

    ---

    Bir kullanıcı promptu ve onun fan-out edileceği personalar.

    Personalar tuple olarak saklanır; bu hem sırayı korur (UI'da
    görüntüleme sırası) hem de frozen=True semantiğiyle tutarlıdır.
    """

    prompt: str
    personas: tuple[Persona, ...]

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("PromptRequest.prompt cannot be empty or whitespace")
        if not self.personas:
            raise ValueError("PromptRequest.personas must contain at least one Persona")
        names = [p.name for p in self.personas]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate persona names in request: {names}")


@dataclass(frozen=True, slots=True)
class ModelSuccess:
    """
    Successful model response carrying generated text and metadata.

    Latency is measured at the orchestrator level (wall clock around
    the API call); tokens is the total of prompt + completion as
    reported by the model.

    ---

    Üretilen metni ve metadata'sını taşıyan başarılı model cevabı.

    Gecikme orchestrator seviyesinde ölçülür (API çağrısı etrafında
    duvar saati); token sayısı modelin raporladığı prompt + completion
    toplamıdır.
    """

    persona_name: str
    model_name: str
    latency_ms: float
    text: str
    tokens: int


@dataclass(frozen=True, slots=True)
class ModelFailure:
    """
    Failed model response carrying error classification and message.

    error_type is a short stable label (e.g. "RateLimit", "Timeout",
    "APIError") so the UI can branch on category without parsing the
    error_message string.

    ---

    Hata sınıflandırması ve mesajını taşıyan başarısız model cevabı.

    error_type kısa ve sabit bir etikettir (örn. "RateLimit",
    "Timeout", "APIError"); böylece UI, error_message metnini
    ayrıştırmadan kategoriye göre dallanabilir.
    """

    persona_name: str
    model_name: str
    latency_ms: float
    error_type: str
    error_message: str


# Discriminated union — orchestrator and UI must branch on isinstance.
# Ayrımlı union — orchestrator ve UI isinstance ile dallanmak zorundadır.
type ModelResponse = ModelSuccess | ModelFailure


def response_to_dict(response: ModelResponse) -> dict[str, object]:
    """
    Serialize a ModelResponse to a JSON-ready dict, tagging it with a
    "kind" discriminator ("success" or "failure") so a JSON consumer
    can tell the two shapes apart without inspecting which fields are
    present.

    ---

    Bir ModelResponse'u JSON'a hazır dict'e serialize eder; "kind"
    ayırıcısı ("success" veya "failure") ile etiketler; böylece JSON
    tüketicisi, hangi alanların var olduğuna bakmadan iki şekli
    birbirinden ayırabilir.
    """
    kind = "success" if isinstance(response, ModelSuccess) else "failure"
    return {"kind": kind, **asdict(response)}


# Default personas — three temperature points from precise to creative.
# Varsayılan personalar — kesinlikten yaratıcılığa üç sıcaklık noktası.
CREATIVE: Final = Persona(
    name="creative",
    temperature=0.9,
    system_prompt="Lean into vivid, exploratory framings; favor unexpected angles.",
)
BALANCED: Final = Persona(
    name="balanced",
    temperature=0.5,
    system_prompt=None,
)
PRECISE: Final = Persona(
    name="precise",
    temperature=0.1,
    system_prompt="Be terse and factual. No hedging, no asides.",
)

DEFAULT_PERSONAS: Final[tuple[Persona, ...]] = (CREATIVE, BALANCED, PRECISE)
