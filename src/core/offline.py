from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from src.schemas.response_schema import SupportResponse

from .config import Settings, project_root, resolve_path
from .guardrails import check_input, check_output
from .llm_client import (
    CallMetrics,
    GuardrailBlocked,
    LLMResult,
    estimate_tokens,
)
from .metrics import MetricsEvent, MetricsLogger, utc_timestamp
from .prompt_loader import get_prompt


KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "factura",
        "cargo",
        "cobro",
        "pago",
        "reembolso",
        "suscripcion",
        "suscripción",
        "impuesto",
        "tarjeta",
        "plan",
    ],
    "technical": [
        "error",
        "api",
        "integracion",
        "integración",
        "servicio",
        "rendimiento",
        "latencia",
        "logs",
        "webhook",
        "timeout",
        "bug",
    ],
    "account": [
        "cuenta",
        "contraseña",
        "password",
        "2fa",
        "perfil",
        "permisos",
        "verificacion",
        "verificación",
        "iniciar sesion",
        "iniciar sesión",
        "acceso",
    ],
    "other": [
        "consulta",
        "informacion",
        "información",
        "horario",
        "soporte",
        "planes",
        "comentario",
        "capacitación",
    ],
}

ANSWERS: dict[str, str] = {
    "billing": (
        "Revisaremos la factura y los cargos del periodo de tu suscripción, incluido el método de pago y la tarjeta registrada. "
        "Si hay duplicidad, gestionaremos el reembolso o ajuste y verificaremos los impuestos."
    ),
    "technical": (
        "Revisaremos los logs del servicio, el error reportado, el timeout y el webhook de la integración o API, "
        "además del rendimiento. Te confirmaremos la causa y los siguientes pasos."
    ),
    "account": (
        "Validaremos el estado de tu cuenta y perfil, el acceso y el mecanismo 2FA o contraseña/password. "
        "También revisaremos permisos y verificación antes de solicitar cambios."
    ),
    "other": (
        "Registraremos tu consulta y verificaremos la información disponible con el equipo de soporte. "
        "Te responderemos con los siguientes pasos."
    ),
}

ACTIONS: dict[str, list[str]] = {
    "billing": [
        "Verificar factura y cargos del periodo",
        "Validar duplicidad en la pasarela de pago",
        "Gestionar reembolso o ajuste si corresponde",
    ],
    "technical": [
        "Revisar logs y métricas de rendimiento",
        "Reproducir el error reportado",
        "Escalar a ingeniería si persiste",
    ],
    "account": [
        "Verificar estado de la cuenta",
        "Revisar contraseña, 2FA y permisos",
        "Escalar a seguridad si persiste",
    ],
    "other": [
        "Registrar el caso",
        "Solicitar detalles adicionales si faltan",
        "Derivar al equipo correspondiente",
    ],
}


@dataclass
class OfflineMetricContext:
    request_id: str
    prompt_version: str


KEYWORD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    category: tuple(
        re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", re.IGNORECASE)
        for keyword in keywords
    )
    for category, keywords in KEYWORDS.items()
}


def classify(query: str) -> tuple[str, int]:
    scores = {
        category: sum(1 for pattern in patterns if pattern.search(query))
        for category, patterns in KEYWORD_PATTERNS.items()
    }
    category = max(scores, key=scores.get)
    return (category, scores[category]) if scores[category] else ("other", 0)


class OfflineResponder:
    def __init__(self, settings: Settings, metrics: MetricsLogger | None = None) -> None:
        self.settings = settings
        self.metrics = metrics or MetricsLogger(
            resolve_path(settings.metrics.path, project_root())
        )

    def generate(self, query: str, prompt_version: str | None = None) -> LLMResult:
        version = prompt_version or self.settings.prompt.version
        get_prompt(version, contract_version=self.settings.version.contract)
        context = OfflineMetricContext(uuid.uuid4().hex[:12], version)
        input_guard = check_input(query, self.settings.guardrails)
        if not input_guard.allowed:
            self._log(
                context,
                status="blocked",
                prompt_tokens=estimate_tokens(query),
                completion_tokens=0,
                cost=0.0,
                error_type="GuardrailBlocked",
            )
            raise GuardrailBlocked(input_guard.reason)

        category, hits = classify(query)
        confidence = 0.86 if hits else 0.42
        if version == "v2_advanced":
            confidence = min(1.0, confidence + 0.03)
        answer = ANSWERS[category]
        if version == "v2_advanced":
            answer += " No compartiré datos sensibles ni información de otras cuentas."
        response = SupportResponse(
            answer=answer,
            confidence=confidence,
            category=category,
            actions=ACTIONS[category],
        )
        output_guard = check_output(response, self.settings.guardrails)
        if not output_guard.allowed or output_guard.response is None:
            self._log(
                context,
                status="error",
                prompt_tokens=estimate_tokens(query),
                completion_tokens=0,
                cost=0.0,
                error_type="GuardrailBlocked",
            )
            raise GuardrailBlocked(output_guard.reason or "Guardrail de salida bloqueó la respuesta")

        prompt_tokens = estimate_tokens(query) + 120
        completion_tokens = estimate_tokens(answer + " ".join(ACTIONS[category]))
        self._log(
            context,
            status="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=0.0,
        )
        return LLMResult(
            response=output_guard.response,
            provider="offline",
            fallback_used=False,
            prompt_version=version,
            request_id=context.request_id,
            metrics=CallMetrics(
                provider="offline",
                model="heuristic-v1",
                api_version="n/a",
                contract_version=self.settings.version.contract,
                prompt_version=version,
                streamed=False,
                latency_seconds=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tokens_source="estimated",
                currency="USD",
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
            ),
            failures=[],
        )

    def _log(
        self,
        context: OfflineMetricContext,
        *,
        status: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        error_type: str | None = None,
    ) -> None:
        self.metrics.log(
            MetricsEvent(
                timestamp=utc_timestamp(),
                request_id=context.request_id,
                provider="offline",
                model="heuristic-v1",
                prompt_version=context.prompt_version,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_seconds=0.0,
                cost_usd=cost,
                currency="USD",
                tokens_source="estimated",
                error_type=error_type,
            )
        )
