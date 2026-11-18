from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class CategoryEnum(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class SupportResponse(BaseModel):
    answer: str = Field(
        description="Respuesta clara y profesional redactada para el cliente final."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Nivel de confianza del modelo en la respuesta, entre 0.0 (sin confianza) y 1.0 (máxima confianza).",
    )
    category: CategoryEnum = Field(
        description="Categoría asignada a la consulta del cliente: facturación, soporte técnico, cuenta u otra."
    )
    actions: List[str] = Field(
        description="Lista de acciones sugeridas para que el agente humano de soporte de nivel 1 continúe o cierre el caso."
    )
