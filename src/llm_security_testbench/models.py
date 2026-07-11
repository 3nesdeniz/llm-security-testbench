"""Core data models and label normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ModelError(ValueError):
    """Raised when an input row or prediction cannot be normalized."""


_POSITIVE_LABELS = {
    "1",
    "attack",
    "blocked",
    "malicious",
    "positive",
    "prompt_injection",
    "unsafe",
    "true",
}
_NEGATIVE_LABELS = {
    "0",
    "allowed",
    "benign",
    "false",
    "negative",
    "safe",
}


def normalize_label(value: Any) -> int:
    """Normalize common binary security labels to ``0`` or ``1``."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in _POSITIVE_LABELS:
            return 1
        if normalized in _NEGATIVE_LABELS:
            return 0
    raise ModelError(f"unsupported binary label: {value!r}")


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelError(f"{field_name} must be a string or null")
    normalized = value.strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class Example:
    """One labeled prompt-injection evaluation example."""

    id: str
    text: str
    label: int
    category: str = "unknown"
    attack_family: str = "none"
    source_context: str = "unknown"
    pair_id: str | None = None
    source_type: str = "unknown"
    split: str = "unspecified"
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Example:
        """Build an example from a JSON-compatible mapping."""

        example_id = payload.get("id")
        text = payload.get("text")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ModelError("example id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ModelError(f"example {example_id!r} has an empty text field")
        if "label" not in payload:
            raise ModelError(f"example {example_id!r} is missing label")

        known_fields = {
            "id",
            "text",
            "label",
            "category",
            "attack_family",
            "source_context",
            "pair_id",
            "source_type",
            "split",
        }
        extra = {key: value for key, value in payload.items() if key not in known_fields}
        return cls(
            id=example_id.strip(),
            text=text,
            label=normalize_label(payload["label"]),
            category=str(payload.get("category", "unknown")),
            attack_family=str(payload.get("attack_family", "none")),
            source_context=str(payload.get("source_context", "unknown")),
            pair_id=_optional_string(payload.get("pair_id"), field_name="pair_id"),
            source_type=str(payload.get("source_type", "unknown")),
            split=str(payload.get("split", "unspecified")),
            extra=extra,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for Python predictors."""

        result: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "category": self.category,
            "attack_family": self.attack_family,
            "source_context": self.source_context,
            "pair_id": self.pair_id,
            "source_type": self.source_type,
            "split": self.split,
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True, slots=True)
class Prediction:
    """A detector prediction for one example."""

    id: str
    label: int | None = None
    score: float | None = None
    latency_ms: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in (0, 1):
            raise ModelError("prediction label must be 0, 1, or null")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ModelError("prediction score must be between 0 and 1")

    @classmethod
    def from_value(
        cls,
        example_id: str,
        value: Any,
        *,
        latency_ms: float | None = None,
    ) -> Prediction:
        """Normalize a Python or HTTP predictor return value."""

        label: int | None = None
        score: float | None = None

        if isinstance(value, Mapping):
            label_value = next(
                (
                    value[key]
                    for key in ("prediction", "predicted_label", "label")
                    if key in value and value[key] is not None
                ),
                None,
            )
            if label_value is not None:
                label = normalize_label(label_value)
            score_value = value.get("score")
            if score_value is not None:
                score = float(score_value)
        elif isinstance(value, float) and not isinstance(value, bool):
            score = value
        else:
            label = normalize_label(value)

        if label is None and score is None:
            raise ModelError("prediction must contain a label, a score, or both")
        return cls(id=example_id, label=label, score=score, latency_ms=latency_ms)

    def resolved_label(self, threshold: float) -> int:
        """Resolve the prediction to a binary label."""

        if self.label is not None:
            return self.label
        if self.score is None:
            raise ModelError(f"prediction {self.id!r} has no label or score")
        return int(self.score >= threshold)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """A labeled example joined with a normalized prediction."""

    example: Example
    prediction: Prediction
    predicted_label: int

    @property
    def correct(self) -> bool:
        return self.example.label == self.predicted_label
