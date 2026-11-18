from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

from src.schemas.response_schema import SupportResponse

from .config import ProviderConfig, Settings, project_root, resolve_path
from .guardrails import check_input, check_output
from .metrics import MetricsEvent, MetricsLogger, utc_timestamp
from .prompt_loader import get_prompt

LOGGER = logging.getLogger("support_llm")
INSTRUCTOR_PARSE_RETRIES = 1


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class GuardrailBlocked(Exception):
    pass


class AllProvidersFailedError(Exception):
    def __init__(self, failures: list[AttemptFailure]) -> None:
        self.failures = failures
        detail = "; ".join(f"{item.provider}: {item.error}" for item in failures)
        super().__init__(f"Todos los proveedores fallaron. {detail or 'Sin detalles.'}")


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostBreakdown(BaseModel):
    currency: str
    input_cost: float
    output_cost: float
    total_cost: float


class AttemptFailure(BaseModel):
    provider: str
    attempt: int
    error: str
    status_code: int | None = None
    retryable: bool = False


class CallMetrics(BaseModel):
    provider: str
    model: str
    api_version: str
    contract_version: str
    prompt_version: str
    streamed: bool
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_source: str
    currency: str
    input_cost: float
    output_cost: float
    total_cost: float


class LLMResult(BaseModel):
    response: SupportResponse
    provider: str
    fallback_used: bool
    prompt_version: str
    request_id: str
    metrics: CallMetrics
    failures: list[AttemptFailure] = Field(default_factory=list)


def estimate_tokens(*texts: str) -> int:
    return max(1, sum(len(text) for text in texts) // 4)


def complete_usage(usage: TokenUsage, prompt_text: str, completion_text: str) -> TokenUsage:
    input_tokens = usage.input_tokens or estimate_tokens(prompt_text)
    output_tokens = usage.output_tokens or estimate_tokens(completion_text)
    estimated = usage.estimated or usage.input_tokens == 0 or usage.output_tokens == 0
    return TokenUsage(input_tokens, output_tokens, estimated)


def calculate_cost(usage: TokenUsage, pricing: Any) -> CostBreakdown:
    scale = Decimal("1000")

    def to_money(value: Decimal) -> float:
        return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

    input_cost = to_money(
        Decimal(usage.input_tokens) * Decimal(str(pricing.input_per_1k_tokens)) / scale
    )
    output_cost = to_money(
        Decimal(usage.output_tokens) * Decimal(str(pricing.output_per_1k_tokens)) / scale
    )
    return CostBreakdown(
        currency=pricing.currency,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=to_money(Decimal(str(input_cost)) + Decimal(str(output_cost))),
    )


def extract_usage(raw: Any, provider: str) -> TokenUsage:
    usage = getattr(raw, "usage", None)
    if provider == "openai":
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
    else:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
    return TokenUsage(input_tokens, output_tokens, not (input_tokens and output_tokens))


class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: float) -> None:
        self.rate = float(rate_per_second)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, amount: float) -> None:
        amount = min(max(amount, 1.0), self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity,
                    self.tokens + (now - self.updated_at) * self.rate,
                )
                self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                wait_seconds = (amount - self.tokens) / self.rate
            time.sleep(max(wait_seconds, 0.01))


class RateLimiter:
    def __init__(self, config: Any) -> None:
        self.requests = TokenBucket(
            config.requests_per_minute / 60,
            config.requests_per_minute,
        )
        self.tokens = TokenBucket(
            config.tokens_per_minute / 60,
            config.tokens_per_minute,
        )

    def acquire(self, estimated_tokens: int) -> None:
        self.requests.acquire(1)
        self.tokens.acquire(estimated_tokens)


class ProviderAdapter:
    name = ""

    def __init__(self, config: ProviderConfig, contract_version: str) -> None:
        self.config = config
        self.contract_version = contract_version
        self.api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not self.api_key:
            raise ProviderError(
                self.name,
                f"Falta la variable de entorno {config.api_key_env}",
                retryable=False,
            )
        self.client = self._build_client()
        self.structured = self._build_structured_client()

    def _headers(self) -> dict[str, str]:
        return {
            "X-Contract-Version": self.contract_version,
            "X-Provider-API-Version": self.config.api_version,
            **self.config.headers,
        }

    def _build_client(self) -> Any:
        raise NotImplementedError

    def _build_structured_client(self) -> Any:
        raise NotImplementedError

    def call(self, system_prompt: str, user_prompt: str) -> tuple[SupportResponse, TokenUsage]:
        raise NotImplementedError


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def _build_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
            default_headers=self._headers(),
        )

    def _build_structured_client(self) -> Any:
        return instructor.from_openai(self.client, mode=instructor.Mode.TOOLS)

    def call(self, system_prompt: str, user_prompt: str) -> tuple[SupportResponse, TokenUsage]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = f"{system_prompt}\n{user_prompt}"
        if self.config.streaming:
            kwargs["stream"] = True
            if self.config.include_usage_in_stream:
                kwargs["stream_options"] = {"include_usage": True}
            partials = self.structured.create(
                messages=messages,
                response_model=instructor.Partial[SupportResponse],
                max_retries=INSTRUCTOR_PARSE_RETRIES,
                **kwargs,
            )
            last = None
            for last in partials:
                pass
            if last is None:
                raise ProviderError(self.name, "El stream no devolvió fragmentos")
            response = SupportResponse.model_validate(last.model_dump())
            usage = TokenUsage(estimated=True)
        else:
            response = self.structured.create(
                messages=messages,
                response_model=SupportResponse,
                max_retries=INSTRUCTOR_PARSE_RETRIES,
                **kwargs,
            )
            usage = extract_usage(getattr(response, "_raw_response", None), self.name)
        completion_text = " ".join([response.answer, *response.actions])
        return response, complete_usage(usage, prompt_text, completion_text)


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    def _build_client(self) -> Anthropic:
        headers = {
            "anthropic-version": self.config.api_version,
            **self._headers(),
        }
        return Anthropic(
            api_key=self.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
            default_headers=headers,
        )

    def _build_structured_client(self) -> Any:
        return instructor.from_anthropic(self.client, mode=instructor.Mode.ANTHROPIC_JSON)

    def call(self, system_prompt: str, user_prompt: str) -> tuple[SupportResponse, TokenUsage]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
        }
        prompt_text = f"{system_prompt}\n{user_prompt}"
        if self.config.streaming:
            partials = self.structured.create(
                response_model=instructor.Partial[SupportResponse],
                max_retries=INSTRUCTOR_PARSE_RETRIES,
                **kwargs,
            )
            last = None
            for last in partials:
                pass
            if last is None:
                raise ProviderError(self.name, "El stream no devolvió fragmentos")
            response = SupportResponse.model_validate(last.model_dump())
            usage = TokenUsage(estimated=True)
        else:
            response = self.structured.create(
                response_model=SupportResponse,
                max_retries=INSTRUCTOR_PARSE_RETRIES,
                **kwargs,
            )
            usage = extract_usage(getattr(response, "_raw_response", None), self.name)
        completion_text = " ".join([response.answer, *response.actions])
        return response, complete_usage(usage, prompt_text, completion_text)


class OpenCodeAdapter(ProviderAdapter):
    name = "opencode"

    def __init__(self, config: ProviderConfig, contract_version: str) -> None:
        self.config = config
        self.contract_version = contract_version
        self.api_key = None
        self.binary = os.environ.get("OPENCODE_BIN") or shutil.which("opencode")
        if not self.binary:
            raise ProviderError(
                self.name, "No se encontró el binario de opencode", retryable=False
            )

    def _build_client(self) -> None:
        return None

    def _build_structured_client(self) -> None:
        return None

    def _payload(self, text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ProviderError(self.name, "opencode no devolvió JSON válido", retryable=True)

    def call(self, system_prompt: str, user_prompt: str) -> tuple[SupportResponse, TokenUsage]:
        prompt = (
            f"{system_prompt}\n\n{user_prompt}\n\n"
            "Respondé solo con un objeto JSON válido que cumpla el contrato indicado."
        )
        try:
            completed = subprocess.run(
                [self.binary, "run", "-m", self.config.model, prompt],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                self.name, f"Timeout tras {self.config.timeout_seconds}s", retryable=True
            ) from exc
        except OSError as exc:
            raise ProviderError(
                self.name, f"No se pudo ejecutar opencode: {exc}", retryable=False
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ProviderError(
                self.name,
                detail[-500:] or f"opencode salió con código {completed.returncode}",
                status_code=completed.returncode,
                retryable=False,
            )
        response = SupportResponse.model_validate(self._payload(completed.stdout))
        completion_text = " ".join([response.answer, *response.actions])
        usage = complete_usage(
            TokenUsage(estimated=True), f"{system_prompt}\n{user_prompt}", completion_text
        )
        return response, usage


def build_adapter(name: str, config: ProviderConfig, contract_version: str) -> ProviderAdapter:
    if name == "openai":
        return OpenAIAdapter(config, contract_version)
    if name == "anthropic":
        return AnthropicAdapter(config, contract_version)
    if name == "opencode":
        return OpenCodeAdapter(config, contract_version)
    raise ProviderError(name, f"Proveedor no soportado: {name}", retryable=False)


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(exc: Exception, config: ProviderConfig) -> bool:
    if isinstance(exc, ProviderError):
        return exc.retryable
    status = _status_code(exc)
    if status is not None and status in config.retry.retry_on_status:
        return True
    return type(exc).__name__ in config.retry.retry_on_exceptions


class LLMClient:
    def __init__(self, settings: Settings, metrics: MetricsLogger | None = None) -> None:
        self.settings = settings
        self.metrics = metrics or MetricsLogger(resolve_path(settings.metrics.path, project_root()))
        self.limiters: dict[str, RateLimiter] = {}
        self.adapters: dict[str, ProviderAdapter] = {}

    def _get_adapter(self, name: str, config: ProviderConfig) -> ProviderAdapter:
        if name not in self.adapters:
            self.adapters[name] = build_adapter(name, config, self.settings.version.contract)
        return self.adapters[name]

    def _log_attempt(
        self,
        *,
        request_id: str,
        adapter: ProviderAdapter,
        prompt_version: str,
        status: str,
        latency_seconds: float,
        usage: TokenUsage | None,
        prompt_tokens_estimate: int,
        cost: CostBreakdown | None,
        error_type: str | None = None,
    ) -> None:
        usage = usage or TokenUsage(
            input_tokens=prompt_tokens_estimate,
            output_tokens=0,
            estimated=True,
        )
        cost = cost or CostBreakdown(
            currency=adapter.config.pricing.currency,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
        )
        self.metrics.log(
            MetricsEvent(
                timestamp=utc_timestamp(),
                request_id=request_id,
                provider=adapter.name,
                model=adapter.config.model,
                prompt_version=prompt_version,
                status=status,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_seconds=round(latency_seconds, 4),
                cost_usd=cost.total_cost,
                currency=cost.currency,
                tokens_source="estimated" if usage.estimated else "api",
                error_type=error_type,
            )
        )

    def _call_with_retries(
        self,
        adapter: ProviderAdapter,
        limiter: RateLimiter,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        request_id: str,
        failures: list[AttemptFailure],
    ) -> tuple[SupportResponse, TokenUsage, float]:
        config = adapter.config
        prompt_tokens_estimate = estimate_tokens(system_prompt, user_prompt) + config.max_tokens
        for attempt in range(1, config.retry.max_attempts + 1):
            started_at = time.perf_counter()
            usage: TokenUsage | None = None
            try:
                limiter.acquire(prompt_tokens_estimate)
                response, usage = adapter.call(system_prompt, user_prompt)
                guard = check_output(response, self.settings.guardrails)
                if not guard.allowed or guard.response is None:
                    raise GuardrailBlocked(guard.reason or "Guardrail de salida bloqueó la respuesta")
                latency = time.perf_counter() - started_at
                cost = calculate_cost(usage, config.pricing)
                self._log_attempt(
                    request_id=request_id,
                    adapter=adapter,
                    prompt_version=prompt_version,
                    status="success",
                    latency_seconds=latency,
                    usage=usage,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    cost=cost,
                )
                LOGGER.info("%s intento %s exitoso (%.3fs)", adapter.name, attempt, latency)
                return guard.response, usage, latency
            except GuardrailBlocked as exc:
                latency = time.perf_counter() - started_at
                failures.append(
                    AttemptFailure(
                        provider=adapter.name,
                        attempt=attempt,
                        error=str(exc),
                        retryable=False,
                    )
                )
                self._log_attempt(
                    request_id=request_id,
                    adapter=adapter,
                    prompt_version=prompt_version,
                    status="blocked",
                    latency_seconds=latency,
                    usage=usage,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    cost=None,
                    error_type="GuardrailBlocked",
                )
                raise
            except Exception as exc:
                latency = time.perf_counter() - started_at
                status = _status_code(exc)
                retryable = _is_retryable(exc, config)
                failures.append(
                    AttemptFailure(
                        provider=adapter.name,
                        attempt=attempt,
                        error=str(exc),
                        status_code=status,
                        retryable=retryable,
                    )
                )
                self._log_attempt(
                    request_id=request_id,
                    adapter=adapter,
                    prompt_version=prompt_version,
                    status="error",
                    latency_seconds=latency,
                    usage=None,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    cost=None,
                    error_type=type(exc).__name__,
                )
                LOGGER.warning("%s intento %s fallido: %s", adapter.name, attempt, exc)
                if not retryable or attempt == config.retry.max_attempts:
                    raise ProviderError(
                        adapter.name,
                        str(exc),
                        status_code=status,
                        retryable=retryable,
                    ) from exc
                delay = min(
                    config.retry.initial_backoff_seconds
                    * (config.retry.backoff_multiplier ** (attempt - 1)),
                    config.retry.max_backoff_seconds,
                )
                delay += random.uniform(0, config.retry.jitter_seconds)
                time.sleep(delay)
        raise ProviderError(adapter.name, "No se pudo completar la llamada", retryable=False)

    def generate(
        self,
        query: str,
        prompt_version: str | None = None,
        provider_override: str | None = None,
    ) -> LLMResult:
        version = prompt_version or self.settings.prompt.version
        request_id = uuid.uuid4().hex[:12]
        input_guard = check_input(query, self.settings.guardrails)
        if not input_guard.allowed:
            self.metrics.log(
                MetricsEvent(
                    timestamp=utc_timestamp(),
                    request_id=request_id,
                    provider="guardrail",
                    model="input_guardrail",
                    prompt_version=version,
                    status="blocked",
                    prompt_tokens=estimate_tokens(query),
                    completion_tokens=0,
                    total_tokens=estimate_tokens(query),
                    latency_seconds=0.0,
                    cost_usd=0.0,
                    currency="USD",
                    tokens_source="estimated",
                    error_type="GuardrailBlocked",
                )
            )
            raise GuardrailBlocked(input_guard.reason)

        system_prompt = get_prompt(version, contract_version=self.settings.version.contract)
        failures: list[AttemptFailure] = []
        provider_order = [provider_override or self.settings.default_provider]
        if provider_override is None and self.settings.fallback.enabled:
            provider_order += [
                name
                for name in self.settings.fallback.providers
                if name != self.settings.default_provider
            ]

        for name in dict.fromkeys(provider_order):
            config = self.settings.providers[name]
            if not config.enabled:
                failures.append(
                    AttemptFailure(
                        provider=name,
                        attempt=0,
                        error="Proveedor deshabilitado en la configuración",
                    )
                )
                continue
            try:
                adapter = self._get_adapter(name, config)
            except ProviderError as exc:
                LOGGER.error("%s no disponible: %s", name, exc)
                failures.append(
                    AttemptFailure(
                        provider=name,
                        attempt=0,
                        error=str(exc),
                        status_code=exc.status_code,
                    )
                )
                continue

            limiter = self.limiters.setdefault(name, RateLimiter(config.rate_limit))
            try:
                response, usage, latency = self._call_with_retries(
                    adapter,
                    limiter,
                    system_prompt,
                    query,
                    version,
                    request_id,
                    failures,
                )
            except ProviderError as exc:
                LOGGER.error("%s agotó los reintentos: %s", name, exc)
                continue

            cost = calculate_cost(usage, config.pricing)
            return LLMResult(
                response=response,
                provider=name,
                fallback_used=name != (provider_override or self.settings.default_provider),
                prompt_version=version,
                request_id=request_id,
                metrics=CallMetrics(
                    provider=name,
                    model=config.model,
                    api_version=config.api_version,
                    contract_version=self.settings.version.contract,
                    prompt_version=version,
                    streamed=config.streaming,
                    latency_seconds=round(latency, 4),
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    tokens_source="estimated" if usage.estimated else "api",
                    currency=cost.currency,
                    input_cost=cost.input_cost,
                    output_cost=cost.output_cost,
                    total_cost=cost.total_cost,
                ),
                failures=failures,
            )

        raise AllProvidersFailedError(failures)
