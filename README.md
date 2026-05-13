# Prismatic

Send one prompt to Gemini and see how different sampling temperatures shape the answer — three responses, side by side, in your terminal.

## What it does

The same prompt is dispatched to a single LLM (currently Gemini) at multiple temperatures in parallel. Each response is rendered as its own panel so you can compare how `temperature` shifts the model from precise and terse to vivid and exploratory.

The name is a metaphor: one beam — the prompt — entering a prism becomes multiple colours, each a different answer.

## Demo

```text
$ python main.py "What is the capital of France?"

╭─ Gemini (creative, temp=0.9) ──────────────────────╮
│ The capital is Paris, a city famed for its         │
│ riverside elegance and centuries of layered        │
│ history beneath every street corner.               │
│                                                    │
│ ⏱  1.2s · 📊 142 tokens                            │
╰────────────────────────────────────────────────────╯

╭─ Gemini (balanced, temp=0.5) ──────────────────────╮
│ Paris is the capital of France.                    │
│                                                    │
│ ⏱  0.8s · 📊 98 tokens                             │
╰────────────────────────────────────────────────────╯

╭─ Gemini (precise, temp=0.1) ───────────────────────╮
│ Paris.                                             │
│                                                    │
│ ⏱  0.6s · 📊 12 tokens                             │
╰────────────────────────────────────────────────────╯
```

When a persona fails (rate limit, timeout, auth error, etc.) its panel turns red and the error category is shown in place of the text; the other personas still complete and render.

## Quickstart

### Requirements

- Python 3.12 or newer
- A Gemini API key (get one at https://aistudio.google.com)

### Install

```bash
git clone https://github.com/Ohualtex/Prismatic.git
cd Prismatic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_real_key
```

### Run

```bash
python main.py "What is the capital of France?"
```

## How it works

Three default personas — `creative` (temperature 0.9), `balanced` (0.5), `precise` (0.1) — fire in parallel via `asyncio.gather`. Each call is wrapped in a retry loop with jittered exponential backoff for transient failures (429, 5xx, timeout); permanent failures (4xx, auth) fail fast. Results are gathered in the original persona order regardless of completion order, then rendered as panels — green for success, red for failure.

Both successful and failed responses are first-class data: the orchestrator returns a list of `ModelResponse` values where every element is either a `ModelSuccess` or a `ModelFailure`, never an exception. Callers branch on response type, not try/except.

## Development

```bash
pip install -r requirements-dev.txt

# Run the test suite (no API calls)
pytest

# Format, lint, type-check
ruff format .
ruff check .
mypy --strict .
```

---

# Prismatic (Türkçe)

Tek bir prompt'u Gemini'ye gönderip farklı sıcaklık (temperature) değerlerinin cevabı nasıl şekillendirdiğini görün — üç cevap, yan yana, terminalinizde.

## Ne yapar?

Aynı prompt tek bir LLM'e (şu an Gemini) birden fazla sıcaklık değeri ile paralel gönderilir. Her cevap kendi paneline render edilir; böylece `temperature`'ün modeli kesin ve özlü olmaktan canlı ve keşifçi olmaya nasıl kaydırdığını karşılaştırabilirsiniz.

İsim bir metafor: tek bir ışın — yani prompt — prizmadan geçince birden fazla renge, yani farklı cevaplara ayrılır.

Bir persona başarısız olduğunda (rate limit, timeout, auth hatası vb.) paneli kırmızıya döner ve metin yerine hata kategorisi gösterilir; diğer personalar yine de tamamlanır ve render edilir.

## Hızlı başlangıç

### Gereksinimler

- Python 3.12 veya üstü
- Gemini API anahtarı (https://aistudio.google.com adresinden alın)

### Kurulum

```bash
git clone https://github.com/Ohualtex/Prismatic.git
cd Prismatic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Yapılandırma

```bash
cp .env.example .env
# .env'yi açın ve GEMINI_API_KEY=gerçek_anahtarınız olarak ayarlayın
```

### Çalıştırma

```bash
python main.py "Fransa'nın başkenti nedir?"
```

## Nasıl çalışır?

Üç varsayılan persona — `creative` (sıcaklık 0.9), `balanced` (0.5), `precise` (0.1) — `asyncio.gather` ile paralel çalışır. Her çağrı, geçici hatalar (429, 5xx, timeout) için jitter'lı üstel backoff retry döngüsüyle sarılır; kalıcı hatalar (4xx, auth) hızlıca başarısız olur. Sonuçlar, tamamlanma sırasından bağımsız olarak orijinal persona sırasında toplanır ve panel olarak render edilir — başarı için yeşil, hata için kırmızı.

Başarılı ve başarısız cevaplar birinci sınıf veridir: orchestrator bir `ModelResponse` listesi döner — her eleman ya `ModelSuccess` ya `ModelFailure`, asla exception değil. Çağıran taraf try/except yerine cevap tipi üzerinden dallanır.

## Geliştirme

```bash
pip install -r requirements-dev.txt

# Test suite'i çalıştır (API çağrısı yok)
pytest

# Format, lint, tip kontrolü
ruff format .
ruff check .
mypy --strict .
```
