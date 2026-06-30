"""
Unit tests for core.persona_loader — valid configs round-trip to
Persona tuples, and every file/parse/structural/field error path
raises ValueError with a "persona config" prefix.

These tests touch the filesystem (tmp_path) but make no network call.

---

core.persona_loader birim testleri — geçerli config'ler Persona
tuple'larına dönüşür ve her dosya/ayrıştırma/yapısal/alan hata yolu
"persona config" önekiyle ValueError fırlatır.

Bu testler dosya sistemine dokunur (tmp_path) ama ağ çağrısı yapmaz.
"""

from pathlib import Path

import pytest

from core.persona_loader import load_personas


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "personas.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestValidConfig:
    def test_round_trips_to_personas(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - name: creative
    temperature: 0.9
    system_prompt: "vivid"
  - name: precise
    temperature: 0.1
""",
        )
        result = load_personas(path)

        assert len(result) == 2
        assert result[0].name == "creative"
        assert result[0].temperature == 0.9
        assert result[0].system_prompt == "vivid"

    def test_system_prompt_optional(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - name: precise
    temperature: 0.1
""",
        )
        result = load_personas(path)
        assert result[0].system_prompt is None

    def test_order_preserved(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - name: a
    temperature: 0.1
  - name: b
    temperature: 0.2
  - name: c
    temperature: 0.3
""",
        )
        assert [p.name for p in load_personas(path)] == ["a", "b", "c"]


class TestFileAndParseErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot read file"):
            load_personas(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "personas: [unclosed")
        with pytest.raises(ValueError, match="invalid YAML"):
            load_personas(path)


class TestStructuralErrors:
    def test_top_level_not_mapping(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "- a\n- b\n")
        with pytest.raises(ValueError, match="top level must be a mapping"):
            load_personas(path)

    def test_missing_personas_key(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "other: stuff\n")
        with pytest.raises(ValueError, match="missing required 'personas'"):
            load_personas(path)

    def test_personas_not_list(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "personas: not_a_list\n")
        with pytest.raises(ValueError, match="'personas' must be a list"):
            load_personas(path)

    def test_entry_not_mapping(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "personas:\n  - just_a_string\n")
        with pytest.raises(ValueError, match=r"personas\[0\] must be a mapping"):
            load_personas(path)


class TestFieldErrors:
    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - name: x
    temperature: 0.5
    sytem_prompt: "typo in key"
""",
        )
        with pytest.raises(ValueError, match="unknown keys"):
            load_personas(path)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - temperature: 0.5
""",
        )
        with pytest.raises(ValueError, match=r"personas\[0\]"):
            load_personas(path)

    def test_temperature_validation_delegated(self, tmp_path: Path) -> None:
        # Persona.__post_init__ rejects the out-of-range value.
        # Persona.__post_init__ aralık dışı değeri reddeder.
        path = _write(
            tmp_path,
            """
personas:
  - name: x
    temperature: 5.0
""",
        )
        with pytest.raises(ValueError, match="outside"):
            load_personas(path)

    def test_empty_name_delegated(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
personas:
  - name: ""
    temperature: 0.5
""",
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            load_personas(path)
