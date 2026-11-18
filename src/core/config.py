from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigurationError(Exception):
    pass


class RateLimitConfig(BaseModel):
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)


class PricingConfig(BaseModel):
    currency: str = Field(min_length=1)
    input_per_1k_tokens: float = Field(ge=0)
    output_per_1k_tokens: float = Field(ge=0)


class RetryConfig(BaseModel):
    max_attempts: int = Field(ge=1)
    initial_backoff_seconds: float = Field(ge=0)
    max_backoff_seconds: float = Field(ge=0)
    backoff_multiplier: float = Field(ge=1)
    jitter_seconds: float = Field(ge=0)
    retry_on_status: list[int] = Field(default_factory=list)
    retry_on_exceptions: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    enabled: bool = True
    api_key_env: str | None = None
    model: str = Field(min_length=1)
    api_version: str = Field(min_length=1)
    base_url: str | None = None
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(ge=0, le=1)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    streaming: bool = False
    include_usage_in_stream: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    rate_limit: RateLimitConfig
    pricing: PricingConfig
    retry: RetryConfig


class VersionConfig(BaseModel):
    contract: str = Field(min_length=1)


class PromptConfig(BaseModel):
    version: str = "v1_zero_shot"


class MetricsConfig(BaseModel):
    path: str = "data/metrics.jsonl"


class EvalsConfig(BaseModel):
    min_score_by_category: dict[str, float] = Field(default_factory=dict)


class GuardrailConfig(BaseModel):
    input_enabled: bool = True
    output_enabled: bool = True
    max_input_chars: int = Field(default=4000, gt=0)
    max_output_chars: int = Field(default=4000, gt=0)
    injection_patterns: list[str] = Field(default_factory=list)
    output_block_patterns: list[str] = Field(default_factory=list)


class FallbackConfig(BaseModel):
    enabled: bool = True
    providers: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    version: VersionConfig
    default_provider: str
    fallback: FallbackConfig
    prompt: PromptConfig
    metrics: MetricsConfig
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    guardrails: GuardrailConfig
    providers: dict[str, ProviderConfig]
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_settings(self) -> Settings:
        if not self.providers:
            raise ValueError("Debe configurar al menos un proveedor.")
        if self.default_provider not in self.providers:
            raise ValueError(f"default_provider desconocido: {self.default_provider}")
        for name in self.fallback.providers:
            if name not in self.providers:
                raise ValueError(f"Proveedor de fallback desconocido: {name}")
        for category, threshold in self.evals.min_score_by_category.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"Umbral inválido para {category}: {threshold}")
        if getattr(logging, self.log_level.upper(), None) is None:
            raise ValueError(f"log_level inválido: {self.log_level}")
        return self


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw: str | Path, preferred: Path | None = None) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    bases = [b for b in (preferred, Path.cwd(), project_root()) if b is not None]
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    return (bases[-1] / candidate).resolve()


def resolve_config_path(path: str | Path | None) -> Path:
    raw = path or os.environ.get("CONFIG_PATH")
    if raw is None:
        return project_root() / "config.yaml"
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), project_root()):
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    return (project_root() / candidate).resolve()


def load_settings(path: str | Path | None = None) -> Settings:
    load_dotenv(project_root() / ".env")
    load_dotenv()
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"No se encontró el archivo de configuración: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML inválido en {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("La configuración raíz debe ser un objeto YAML.")

    raw["log_level"] = os.environ.get("LOG_LEVEL", raw.get("log_level", "INFO"))
    raw.setdefault("prompt", {})["version"] = os.environ.get(
        "PROMPT_VERSION", raw["prompt"].get("version", "v1_zero_shot")
    )
    raw.setdefault("metrics", {})["path"] = os.environ.get(
        "METRICS_PATH", raw["metrics"].get("path", "data/metrics.jsonl")
    )
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Configuración inválida en {config_path}: {exc}") from exc
