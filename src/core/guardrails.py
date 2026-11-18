from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from pydantic import ValidationError

from src.schemas.response_schema import SupportResponse

from .config import GuardrailConfig


@dataclass(frozen=True)
class InputGuardResult:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class OutputGuardResult:
    allowed: bool
    response: SupportResponse | None = None
    reason: str = ""


DEFAULT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?",
    r"ignora\s+(todas?\s+)?(las\s+)?instrucciones",
    r"olvida\s+(tus|las)\s+instrucciones",
    r"revela\s+(tu|el)\s+(system\s+)?prompt",
    r"muestra\s+(tu|el)\s+(system\s+)?prompt",
    r"system\s+prompt",
    r"prompt\s+de\s+sistema",
    r"you\s+are\s+now",
    r"act\s+as\s+(an?\s+)?(admin|developer|system)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"<\|\s*(system|assistant|user)\s*\|>",
]

DEFAULT_OUTPUT_BLOCK_PATTERNS = [
    r"system\s+prompt",
    r"prompt\s+de\s+sistema",
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions?",
    r"<\|\s*(system|assistant|user)\s*\|>",
    r"api[_ -]?key\s*[:=]?\s*\S+",
    r"sk-[A-Za-z0-9]{16,}",
]


@lru_cache(maxsize=32)
def _compile(patterns: tuple[str, ...], defaults: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    source = dict.fromkeys(patterns + defaults)
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in source)


def check_input(query: str, config: GuardrailConfig) -> InputGuardResult:
    if not config.input_enabled:
        return InputGuardResult(True)
    if not query or not query.strip():
        return InputGuardResult(False, "Consulta vacía")
    if len(query) > config.max_input_chars:
        return InputGuardResult(False, "Consulta demasiado larga")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in query):
        return InputGuardResult(False, "Caracteres de control no permitidos")
    for pattern in _compile(tuple(config.injection_patterns), tuple(DEFAULT_INJECTION_PATTERNS)):
        if pattern.search(query):
            return InputGuardResult(False, "Posible prompt injection detectado")
    return InputGuardResult(True)


def check_output(
    response: SupportResponse,
    config: GuardrailConfig,
) -> OutputGuardResult:
    if not config.output_enabled:
        return OutputGuardResult(True, response)
    try:
        validated = SupportResponse.model_validate(response.model_dump())
    except ValidationError as exc:
        return OutputGuardResult(False, None, f"Contrato inválido: {exc}")

    answer = validated.answer.strip()
    actions = [action.strip() for action in validated.actions if action.strip()]
    if not answer:
        return OutputGuardResult(False, None, "Respuesta vacía")
    if len(answer) > config.max_output_chars:
        return OutputGuardResult(False, None, "Respuesta demasiado larga")

    text = "\n".join([answer, *actions])
    for pattern in _compile(tuple(config.output_block_patterns), tuple(DEFAULT_OUTPUT_BLOCK_PATTERNS)):
        if pattern.search(text):
            return OutputGuardResult(False, None, "Contenido sensible o sospechoso detectado")

    sanitized = SupportResponse(
        answer=answer,
        confidence=validated.confidence,
        category=validated.category,
        actions=actions,
    )
    return OutputGuardResult(True, sanitized)
