# Reporte de decisión de evals

- Modo: `offline`
- Proveedor: `n/a`
- Dataset: 36 casos

## Regla de decisión

- Acierto = `category` con exact match; `keyword_coverage` se reporta aparte.
- Una versión queda descartada si cualquier categoría cae bajo su umbral.
- Costo por acierto = costo total / aciertos sobre casos con respuesta válida.

## Resumen

| Prompt | Score | Accuracy | Keywords | Umbrales | Costo total | Costo/acierto |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `v1_zero_shot` | 0.9583 | 0.9722 | 0.9259 | Bajo: other (0.8889 < 0.90) | $0.000000 | 0.0 |
| `v2_advanced` | 0.9583 | 0.9722 | 0.9259 | Bajo: other (0.8889 < 0.90) | $0.000000 | 0.0 |

## Por categoría

| Prompt | Categoría | Score | Umbral | Estado |
| --- | --- | ---: | ---: | --- |
| `v1_zero_shot` | account | 1.0000 | 0.90 | OK |
| `v1_zero_shot` | billing | 0.9611 | 0.90 | OK |
| `v1_zero_shot` | other | 0.8889 | 0.90 | DESCARTADO |
| `v1_zero_shot` | technical | 0.9833 | 0.90 | OK |
| `v2_advanced` | account | 1.0000 | 0.90 | OK |
| `v2_advanced` | billing | 0.9611 | 0.90 | OK |
| `v2_advanced` | other | 0.8889 | 0.90 | DESCARTADO |
| `v2_advanced` | technical | 0.9833 | 0.90 | OK |

## Adversarial

| Prompt | Mitigados |
| --- | ---: |
| `v1_zero_shot` | 10/10 |
| `v2_advanced` | 10/10 |

## Decisión

- Mejor score: `v1_zero_shot` (0.9583).
- Mejor costo por acierto: `v1_zero_shot` (0.0).
- ¿Misma versión?: sí.
- Costos reales disponibles: no.
- Recomendación: No promover automáticamente: mejor score incumple umbrales en other.
