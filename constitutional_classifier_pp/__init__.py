"""
Constitutional Classifier++: Two-stage jailbreak defense system.

A production-grade defense system that uses:
- Stage 1: Linear activation probe for fast screening
- Stage 2: External LLM classifier for escalated traffic
- Ensemble scoring for final refusal decisions
"""

from .config.base import ProbeConfig, ClassifierConfig, EnsembleConfig
from .models.probe import LinearActivationProbe
from .models.classifier import ExternalClassifier
from .models.ensemble import TwoStageEnsemble, EnsembleResult
from .hooks.activation_collector import MultiLayerActivationCollector
from .inference.pipeline import ConstitutionalClassifierPipeline

__version__ = "0.1.0"

__all__ = [
    "ProbeConfig",
    "ClassifierConfig",
    "EnsembleConfig",
    "LinearActivationProbe",
    "ExternalClassifier",
    "TwoStageEnsemble",
    "EnsembleResult",
    "MultiLayerActivationCollector",
    "ConstitutionalClassifierPipeline",
]
