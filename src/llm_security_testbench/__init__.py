"""LLM Security Testbench public package API."""

__version__ = "0.1.0"

from llm_security_testbench.evaluator import EvaluationResult, evaluate
from llm_security_testbench.models import Example, Prediction

__all__ = ["EvaluationResult", "Example", "Prediction", "evaluate"]
