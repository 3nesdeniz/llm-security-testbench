"""Prediction adapters for offline files, Python callables, and HTTP endpoints."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from llm_security_testbench.models import Example, ModelError, Prediction


class PredictorError(RuntimeError):
    """Raised when a predictor cannot be initialized or executed."""


class Predictor(Protocol):
    """Common interface implemented by all prediction adapters."""

    name: str

    def predict_many(
        self,
        examples: Sequence[Example],
        *,
        max_workers: int = 1,
    ) -> dict[str, Prediction]:
        """Predict labels or scores for a sequence of examples."""


def _prediction_error(example_id: str, error: Exception | str) -> Prediction:
    return Prediction(id=example_id, error=str(error))


def _parallel_predictions(
    examples: Sequence[Example],
    function: Callable[[Example], Prediction],
    *,
    max_workers: int,
) -> dict[str, Prediction]:
    if max_workers < 1:
        raise PredictorError("max_workers must be at least 1")
    if max_workers == 1:
        return {example.id: function(example) for example in examples}

    results: dict[str, Prediction] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(function, example): example.id for example in examples}
        for future in as_completed(futures):
            example_id = futures[future]
            try:
                results[example_id] = future.result()
            except Exception as exc:  # defensive boundary around third-party predictors
                results[example_id] = _prediction_error(example_id, exc)
    return results


class JsonlPredictor:
    """Read precomputed predictions from a JSONL file."""

    name = "offline-jsonl"

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._predictions = self._load()

    def _load(self) -> dict[str, Prediction]:
        predictions: dict[str, Prediction] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PredictorError(f"could not read predictions file {self.path}: {exc}") from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictorError(
                    f"{self.path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise PredictorError(f"{self.path}:{line_number}: row must be an object")
            example_id = payload.get("id")
            if not isinstance(example_id, str) or not example_id:
                raise PredictorError(f"{self.path}:{line_number}: missing id")
            if example_id in predictions:
                raise PredictorError(f"{self.path}:{line_number}: duplicate id {example_id!r}")

            normalized_payload = dict(payload)
            if (
                "prediction" not in normalized_payload
                and "predicted_label" not in normalized_payload
                and "score" not in normalized_payload
            ):
                raise PredictorError(
                    f"{self.path}:{line_number}: expected prediction, predicted_label, or score"
                )
            try:
                predictions[example_id] = Prediction.from_value(example_id, normalized_payload)
            except (ModelError, TypeError, ValueError) as exc:
                raise PredictorError(f"{self.path}:{line_number}: {exc}") from exc
        return predictions

    def predict_many(
        self,
        examples: Sequence[Example],
        *,
        max_workers: int = 1,
    ) -> dict[str, Prediction]:
        del max_workers
        return {
            example.id: self._predictions.get(
                example.id,
                Prediction(id=example.id, error="missing prediction"),
            )
            for example in examples
        }


def _load_module(spec: str) -> tuple[ModuleType, str]:
    if ":" not in spec:
        raise PredictorError("Python predictors must use path.py:function or module:function")
    module_spec, function_name = spec.rsplit(":", 1)
    path = Path(module_spec).expanduser()
    if path.suffix == ".py" or path.exists():
        resolved = path.resolve()
        if not resolved.is_file():
            raise PredictorError(f"Python predictor file does not exist: {resolved}")
        module_name = f"llmst_user_{resolved.stem}_{abs(hash(resolved))}"
        import_spec = importlib.util.spec_from_file_location(module_name, resolved)
        if import_spec is None or import_spec.loader is None:
            raise PredictorError(f"could not import Python predictor: {resolved}")
        module = importlib.util.module_from_spec(import_spec)
        import_spec.loader.exec_module(module)
        return module, function_name
    return importlib.import_module(module_spec), function_name


class PythonPredictor:
    """Call a user-provided Python function for each example."""

    def __init__(self, spec: str) -> None:
        module, function_name = _load_module(spec)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise PredictorError(f"Python predictor function not found: {spec}")
        self.function: Callable[[dict[str, Any]], Any] = function
        self.name = f"python:{spec}"

    def _predict(self, example: Example) -> Prediction:
        started = time.perf_counter()
        try:
            value = self.function(example.to_mapping())
            latency_ms = (time.perf_counter() - started) * 1000
            return Prediction.from_value(example.id, value, latency_ms=latency_ms)
        except Exception as exc:  # user code is an isolation boundary
            return _prediction_error(example.id, exc)

    def predict_many(
        self,
        examples: Sequence[Example],
        *,
        max_workers: int = 1,
    ) -> dict[str, Prediction]:
        return _parallel_predictions(examples, self._predict, max_workers=max_workers)


_MISSING = object()


def _nested_value(payload: Mapping[str, Any], path: str | None) -> Any:
    if not path:
        return _MISSING
    current: Any = payload
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


class HttpPredictor:
    """POST examples to a JSON detector endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        label_field: str = "label",
        score_field: str = "score",
        text_field: str = "text",
        timeout: float = 15.0,
        retries: int = 1,
        token_env: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise PredictorError("HTTP predictor endpoint must start with http:// or https://")
        if timeout <= 0:
            raise PredictorError("HTTP timeout must be positive")
        if retries < 0:
            raise PredictorError("HTTP retries cannot be negative")

        self.endpoint = endpoint
        self.label_field = label_field
        self.score_field = score_field
        self.text_field = text_field
        self.timeout = timeout
        self.retries = retries
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "llm-security-testbench/0.1.0",
            **dict(headers or {}),
        }
        if token_env:
            token = os.environ.get(token_env)
            if not token:
                raise PredictorError(f"token environment variable is not set: {token_env}")
            self.headers["Authorization"] = f"Bearer {token}"
        self.name = f"http:{endpoint}"

    def _predict(self, example: Example) -> Prediction:
        payload = {
            self.text_field: example.text,
            "id": example.id,
            "metadata": {
                "category": example.category,
                "attack_family": example.attack_family,
                "source_context": example.source_context,
                "pair_id": example.pair_id,
                "split": example.split,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers=self.headers,
                method="POST",
            )
            try:
                # The constructor rejects every scheme except HTTP and HTTPS.
                with urllib.request.urlopen(  # nosec B310
                    request, timeout=self.timeout
                ) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(response_payload, Mapping):
                    raise PredictorError("HTTP response must be a JSON object")
                label = _nested_value(response_payload, self.label_field)
                score = _nested_value(response_payload, self.score_field)
                normalized: dict[str, Any] = {}
                if label is not _MISSING:
                    normalized["prediction"] = label
                if score is not _MISSING:
                    normalized["score"] = score
                latency_ms = (time.perf_counter() - started) * 1000
                return Prediction.from_value(example.id, normalized, latency_ms=latency_ms)
            except (
                json.JSONDecodeError,
                ModelError,
                PredictorError,
                TypeError,
                ValueError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))

        return _prediction_error(example.id, last_error or "unknown HTTP predictor error")

    def predict_many(
        self,
        examples: Sequence[Example],
        *,
        max_workers: int = 1,
    ) -> dict[str, Prediction]:
        return _parallel_predictions(examples, self._predict, max_workers=max_workers)
