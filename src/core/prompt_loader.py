from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .config import project_root


class PromptSpec(BaseModel):
    name: str
    technique: str = ""
    description: str = ""
    comparison_notes: str = ""
    system_prompt: str = Field(min_length=1)


class PromptNotFoundError(KeyError):
    pass


def resolve_prompt_dir() -> Path:
    raw = os.environ.get("PROMPTS_DIR")
    if not raw:
        return project_root() / "prompts"
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), project_root()):
        resolved = base / candidate
        if resolved.is_dir():
            return resolved.resolve()
    return (project_root() / candidate).resolve()


@lru_cache(maxsize=32)
def load_prompt_spec(version: str) -> PromptSpec:
    prompt_dir = resolve_prompt_dir()
    name = Path(version).stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise PromptNotFoundError(f"Versión de prompt inválida: {version}")
    path = prompt_dir / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in prompt_dir.glob("*.yaml")))
        raise PromptNotFoundError(
            f"No existe el prompt '{version}' en {prompt_dir}. Disponibles: {available or 'ninguno'}"
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML inválido en {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"El prompt {path} debe ser un objeto YAML.")
    raw.setdefault("name", name)
    return PromptSpec.model_validate(raw)


def get_prompt(version: str, **values: str) -> str:
    prompt = load_prompt_spec(version).system_prompt
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt.strip()


def list_prompts() -> list[dict[str, str]]:
    prompt_dir = resolve_prompt_dir()
    specs = []
    for path in sorted(prompt_dir.glob("*.yaml")):
        spec = load_prompt_spec(path.stem)
        specs.append(
            {
                "version": spec.name,
                "technique": spec.technique,
                "description": spec.description,
                "comparison_notes": spec.comparison_notes,
            }
        )
    return specs
