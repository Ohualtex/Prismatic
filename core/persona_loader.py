"""
Load a tuple of Persona objects from a YAML config file so users can
define their own personas without editing code.

The expected shape is a top-level "personas" key holding a list of
mappings, each with name, temperature, and an optional system_prompt:

    personas:
      - name: creative
        temperature: 0.9
        system_prompt: "Lean into vivid framings."
      - name: precise
        temperature: 0.1

Field-level validation (temperature range, empty name) is delegated to
Persona.__post_init__; this loader only handles file, parse, and
structural errors, normalising them all into ValueError with a
"persona config" prefix so the source is obvious.

---

Kullanıcıların koda dokunmadan kendi personalarını tanımlayabilmesi
için bir YAML config dosyasından Persona tuple'ı yükler.

Beklenen şekil, her biri name, temperature ve opsiyonel system_prompt
taşıyan eşlemelerden oluşan bir liste tutan top-level "personas"
anahtarıdır (yukarıdaki örneğe bakın).

Alan düzeyi doğrulama (sıcaklık aralığı, boş ad) Persona.__post_init__'e
delege edilir; bu loader yalnızca dosya, ayrıştırma ve yapısal hataları
ele alır ve hepsini "persona config" önekiyle ValueError'a normalize
eder; böylece kaynak bellidir.
"""

from pathlib import Path

import yaml

from core.models import Persona

# Keys a persona mapping may contain. Extras are rejected to catch typos.
# Bir persona eşlemesinin içerebileceği anahtarlar. Fazlalıklar typo
# yakalamak için reddedilir.
_ALLOWED_KEYS = {"name", "temperature", "system_prompt"}


def load_personas(path: Path) -> tuple[Persona, ...]:
    """
    Read, parse, and validate the persona config at path. Returns a
    tuple of Persona in file order. Raises ValueError (with a "persona
    config" prefix) for any file, YAML, or structural problem.

    ---

    path'teki persona config'i okur, ayrıştırır ve doğrular. Dosya
    sırasında bir Persona tuple'ı döner. Herhangi bir dosya, YAML veya
    yapısal sorun için ("persona config" önekiyle) ValueError fırlatır.
    """
    prefix = f"persona config ({path})"

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{prefix}: cannot read file — {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{prefix}: invalid YAML — {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{prefix}: top level must be a mapping with a 'personas' key")
    if "personas" not in data:
        raise ValueError(f"{prefix}: missing required 'personas' key")

    entries = data["personas"]
    if not isinstance(entries, list):
        raise ValueError(f"{prefix}: 'personas' must be a list")

    personas: list[Persona] = []
    for index, entry in enumerate(entries):
        location = f"{prefix}: personas[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location} must be a mapping")

        unknown = set(entry) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{location} has unknown keys: {sorted(unknown)}; "
                f"allowed keys are {sorted(_ALLOWED_KEYS)}"
            )

        try:
            personas.append(Persona(**entry))
        except (TypeError, ValueError) as exc:
            # TypeError: missing required field; ValueError: failed validation.
            # TypeError: eksik zorunlu alan; ValueError: doğrulama başarısız.
            raise ValueError(f"{location}: {exc}") from exc

    return tuple(personas)
