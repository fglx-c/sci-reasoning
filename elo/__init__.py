"""
Simple ELO evaluation system adapted from CoI-Agent.
"""

from .evaluator import (
    ELOEvaluator,
    EvaluationItem,
    ComparisonResult,
    ELOResult,
    evaluate_ideas
)
from .config import (
    ELOConfig,
    DEFAULT_CONFIG
)

__all__ = [
    "ELOEvaluator",
    "EvaluationItem", 
    "ComparisonResult",
    "ELOResult",
    "evaluate_ideas",
    "ELOConfig",
    "DEFAULT_CONFIG"
] 