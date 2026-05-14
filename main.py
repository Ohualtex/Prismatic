"""
Command-line entry point for Prismatic.

Loads GEMINI_API_KEY from .env, dispatches the user prompt to the
three default personas in parallel through the Orchestrator, and
renders the results as Rich panels in input persona order.

---

Prismatic için komut satırı giriş noktası.

GEMINI_API_KEY .env'den yüklenir, kullanıcı promptu Orchestrator
aracılığıyla üç default personaya paralel olarak dağıtılır ve
sonuçlar girdi persona sırasında Rich panelleri olarak render edilir.
"""

import argparse
import asyncio
import os
import sys
from typing import Final

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from core.gemini_client import GeminiClient
from core.models import (
    DEFAULT_PERSONAS,
    ModelFailure,
    ModelResponse,
    ModelSuccess,
    Persona,
    PromptRequest,
)
from core.orchestrator import Orchestrator

# Placeholder value shipped in .env.example. Keep in sync with that file.
# .env.example'da gönderilen yer tutucu değer. O dosya ile senkron tutulmalı.
_PLACEHOLDER_API_KEY: Final = "paste_your_real_key_here"


def _api_key_setup_error() -> str | None:
    """
    Return a user-facing message if GEMINI_API_KEY is missing or still
    the placeholder value; otherwise None. Catching the placeholder
    case here keeps the user from seeing a verbose 400 INVALID_ARGUMENT
    dump from the API for an obviously-unconfigured key.

    ---

    GEMINI_API_KEY eksikse veya hâlâ yer tutucu değerdeyse kullanıcıya
    yönelik mesaj döner; aksi halde None. Yer tutucu durumunu burada
    yakalamak, açıkça yapılandırılmamış bir anahtar için kullanıcının
    API'den ayrıntılı 400 INVALID_ARGUMENT dökümü görmesini önler.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return (
            "GEMINI_API_KEY is not set. Copy .env.example to .env and "
            "add a real key from https://aistudio.google.com"
        )
    if key == _PLACEHOLDER_API_KEY:
        return (
            "GEMINI_API_KEY is still the placeholder value. Edit .env "
            "and paste a real key from https://aistudio.google.com"
        )
    return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prismatic",
        description=("Send one prompt to Gemini across multiple personas in parallel."),
    )
    parser.add_argument(
        "prompt",
        help="The user prompt to dispatch to every persona.",
    )
    return parser


def _render_response(persona: Persona, response: ModelResponse) -> Panel:
    """
    Build a Rich Panel from a single (persona, response) pair.

    Successful panels carry a green border and the generated text;
    failed panels are red and display the error_type label followed
    by the message. Both footers include latency in seconds, and
    success additionally includes the token count.

    ---

    Tek bir (persona, response) ikilisinden Rich Panel oluşturur.

    Başarılı paneller yeşil kenarlı ve üretilen metni taşır;
    başarısız paneller kırmızı kenarlı ve error_type etiketinin
    ardından mesajı gösterir. İki tip de altbilgide gecikmeyi
    saniye olarak içerir; başarı durumunda ek olarak token sayısı
    da yer alır.
    """
    title = f"Gemini ({persona.name}, temp={persona.temperature})"

    if isinstance(response, ModelSuccess):
        body: Text = Text(response.text)
        metrics = Text(
            f"⏱  {response.latency_ms / 1000:.1f}s · 📊 {response.tokens} tokens",
            style="dim",
        )
        border_style = "green"
    else:
        body = Text()
        body.append(response.error_type, style="bold red")
        body.append("\n")
        body.append(response.error_message)
        metrics = Text(
            f"⏱  {response.latency_ms / 1000:.1f}s · failed",
            style="dim",
        )
        border_style = "red"

    content = Group(body, Text(), metrics)
    return Panel(content, title=title, title_align="left", border_style=border_style)


async def _amain(prompt: str) -> int:
    """
    Async entry: orchestrate the fan-out and render results.

    Returns 0 if every persona produced a ModelSuccess, 1 if any
    failed. The CLI wrapper translates these into process exit codes.

    ---

    Async giriş: fan-out'u orchestrate eder ve sonuçları render eder.

    Her persona ModelSuccess ürettiyse 0, herhangi biri başarısız
    olduysa 1 döner. CLI sarmalayıcısı bunları işlem çıkış kodlarına
    çevirir.
    """
    client = GeminiClient()
    orchestrator = Orchestrator(client)
    request = PromptRequest(prompt=prompt, personas=DEFAULT_PERSONAS)
    responses = await orchestrator.run(request)

    console = Console()
    for i, (persona, response) in enumerate(
        zip(DEFAULT_PERSONAS, responses, strict=True)
    ):
        if i > 0:
            console.print()
        console.print(_render_response(persona, response))

    return 1 if any(isinstance(r, ModelFailure) for r in responses) else 0


def main() -> int:
    """
    Synchronous CLI entry point. Translates argparse arguments and
    setup failures into appropriate process exit codes.

    ---

    Senkron CLI giriş noktası. argparse argümanlarını ve kurulum
    hatalarını uygun işlem çıkış kodlarına çevirir.
    """
    args = _build_argparser().parse_args()
    load_dotenv()

    setup_error = _api_key_setup_error()
    if setup_error is not None:
        print(f"Error: {setup_error}", file=sys.stderr)
        return 2

    try:
        return asyncio.run(_amain(args.prompt))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        # Surface other validation errors (e.g. negative retries) to stderr.
        # Diğer doğrulama hatalarını (örn. negatif retry) stderr'e yansıt.
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
