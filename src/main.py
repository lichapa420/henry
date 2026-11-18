from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ConfigurationError, load_settings
from src.core.llm_client import AllProvidersFailedError, GuardrailBlocked, LLMClient
from src.core.offline import OfflineResponder
from src.core.prompt_loader import PromptNotFoundError, list_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI de triage: devuelve un JSON validado con el contrato de soporte."
    )
    parser.add_argument("query", nargs="?", help="Consulta del cliente a triagear.")
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Versión del prompt en prompts/*.yaml (default: v1_zero_shot).",
    )
    parser.add_argument("--provider", default=None, help="Fuerza un proveedor configurado.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usa respondedor heurístico local sin APIs (smoke test).",
    )
    parser.add_argument("--full", action="store_true", help="Imprime métricas junto a la respuesta.")
    parser.add_argument("--list-prompts", action="store_true", help="Lista prompts disponibles.")
    parser.add_argument("--verbose", action="store_true", help="Habilita logs por stderr.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    level = logging.DEBUG if args.verbose else getattr(logging, settings.log_level.upper(), logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.list_prompts:
        for item in list_prompts():
            print(f"{item['version']} | {item['technique']} | {item['description']}")
        return 0
    if not args.query:
        parser.error("Debes pasar una consulta o usar --list-prompts")
    if args.offline and args.provider:
        parser.error("--offline y --provider son mutually exclusivos")

    version = args.prompt_version or settings.prompt.version
    try:
        if args.offline:
            result = OfflineResponder(settings).generate(args.query, version)
        else:
            result = LLMClient(settings).generate(args.query, version, args.provider)
    except GuardrailBlocked as exc:
        print(f"Guardrail bloqueó la request: {exc}", file=sys.stderr)
        return 3
    except (AllProvidersFailedError, ConfigurationError, PromptNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    payload = result if args.full else result.response
    print(payload.model_dump_json(indent=2, ensure_ascii=False))
    if not args.full:
        print(
            f"provider={result.provider} prompt={result.prompt_version} "
            f"tokens={result.metrics.total_tokens} cost_usd={result.metrics.total_cost:.6f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
