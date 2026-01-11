"""Two-Stage Ensemble for Constitutional Classifier++.

This module orchestrates the cascade between Stage 1 (linear probe) and
Stage 2 (external classifier), implementing the ensemble scoring formula
and threshold calibration.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config.base import EnsembleConfig
from models.probe import LinearActivationProbe
from models.classifier import ExternalClassifier


@dataclass
class EnsembleResult:
    """Result from two-stage ensemble classification."""

    stage1_score: float  # Probe probability
    stage2_score: Optional[float]  # Classifier probability (if escalated)
    final_score: float  # Ensemble weighted score
    escalated: bool  # Whether Stage 2 was invoked
    should_refuse: bool  # Final refusal decision

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "stage1_score": self.stage1_score,
            "stage2_score": self.stage2_score,
            "final_score": self.final_score,
            "escalated": self.escalated,
            "should_refuse": self.should_refuse,
        }


class TwoStageEnsemble:
    """Two-stage ensemble combining linear probe and external classifier.

    Stage 1: Linear probe screens all traffic (cheap, fast)
    Stage 2: External classifier processes escalated traffic (~5.5%)

    Ensemble formula: z_final = 0.55 * z_probe + 0.45 * z_classifier

    The ensemble combines complementary signals:
    - Probe: Fast, activation-based, catches obvious patterns
    - Classifier: Slow, context-aware, handles nuanced cases

    Example:
        >>> ensemble = TwoStageEnsemble(probe, classifier, config)
        >>> result = ensemble.classify(activations, user_input, model_output)
        >>> if result.should_refuse:
        ...     return "I cannot help with that request."
    """

    def __init__(
        self,
        probe: LinearActivationProbe,
        classifier: Union[ExternalClassifier, "MockClassifier"],
        config: EnsembleConfig,
    ):
        """Initialize the ensemble.

        Args:
            probe: Stage 1 linear activation probe
            classifier: Stage 2 external classifier
            config: Ensemble configuration
        """
        self.probe = probe
        self.classifier = classifier
        self.config = config

        self.probe_weight = config.probe_weight
        self.classifier_weight = config.classifier_weight
        self.t1_threshold = config.t1_threshold
        self.t2_threshold = config.t2_threshold

        # Statistics tracking
        self._total_classifications = 0
        self._escalations = 0
        self._refusals = 0

    def classify(
        self,
        activations: torch.Tensor,
        user_input: str,
        model_output: str,
        force_stage2: bool = False,
    ) -> EnsembleResult:
        """Run two-stage classification on an exchange.

        Args:
            activations: Concatenated layer activations
                [batch, seq_len, hidden_dim * num_layers] or
                [seq_len, hidden_dim * num_layers]
            user_input: User's message
            model_output: AI's response
            force_stage2: Force escalation to Stage 2 (for testing/debugging)

        Returns:
            EnsembleResult with scores and decision
        """
        # Handle input shapes
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)

        # Stage 1: Linear probe
        with torch.no_grad():
            probe_logits = self.probe(activations)

            # Use max logit across sequence (most suspicious token)
            if self.config.max_probe_score_for_decision:
                max_logit = probe_logits.max(dim=1)[0].squeeze()
            else:
                # Use last token
                max_logit = probe_logits[:, -1, :].squeeze()

            stage1_score = torch.sigmoid(max_logit).item()

        # Track statistics
        self._total_classifications += 1

        # Check if escalation needed
        escalated = stage1_score > self.t1_threshold or force_stage2

        if escalated:
            self._escalations += 1

            # Stage 2: External classifier
            stage2_score = self.classifier.get_probability(user_input, model_output)

            # Ensemble: z_final = probe_weight * z_probe + classifier_weight * z_classifier
            final_score = (
                self.probe_weight * stage1_score +
                self.classifier_weight * stage2_score
            )
        else:
            stage2_score = None
            final_score = stage1_score

        # Final decision
        should_refuse = final_score > self.t2_threshold

        if should_refuse:
            self._refusals += 1

        return EnsembleResult(
            stage1_score=stage1_score,
            stage2_score=stage2_score,
            final_score=final_score,
            escalated=escalated,
            should_refuse=should_refuse,
        )

    def classify_with_probe_only(
        self,
        activations: torch.Tensor,
    ) -> EnsembleResult:
        """Classify using only Stage 1 (probe).

        Useful for fast screening or when classifier is unavailable.

        Args:
            activations: Probe input activations

        Returns:
            EnsembleResult (stage2_score will be None)
        """
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)

        with torch.no_grad():
            probe_logits = self.probe(activations)
            max_logit = probe_logits.max(dim=1)[0].squeeze()
            stage1_score = torch.sigmoid(max_logit).item()

        return EnsembleResult(
            stage1_score=stage1_score,
            stage2_score=None,
            final_score=stage1_score,
            escalated=False,
            should_refuse=stage1_score > self.t2_threshold,
        )

    def calibrate_thresholds(
        self,
        validation_data: List[Tuple[torch.Tensor, str, str, int]],
        target_refusal_rate: Optional[float] = None,
        target_escalation_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calibrate T1 and T2 thresholds on validation data.

        Args:
            validation_data: List of (activations, user_input, model_output, label)
            target_refusal_rate: Target false positive rate (default from config)
            target_escalation_rate: Target escalation rate (default from config)

        Returns:
            Dict with calibrated thresholds and actual rates
        """
        target_refusal_rate = target_refusal_rate or self.config.target_refusal_rate
        target_escalation_rate = target_escalation_rate or self.config.target_escalation_rate

        # Collect Stage 1 scores on all data
        stage1_scores = []
        labels = []

        for acts, user, output, label in validation_data:
            with torch.no_grad():
                if acts.dim() == 2:
                    acts = acts.unsqueeze(0)
                logits = self.probe(acts)
                score = torch.sigmoid(logits.max(dim=1)[0]).item()
                stage1_scores.append(score)
                labels.append(label)

        stage1_scores = np.array(stage1_scores)
        labels = np.array(labels)

        # Calibrate T1: Set so target_escalation_rate of traffic is escalated
        sorted_scores = np.sort(stage1_scores)[::-1]  # Descending
        t1_idx = int(len(sorted_scores) * target_escalation_rate)
        t1_idx = min(t1_idx, len(sorted_scores) - 1)
        self.t1_threshold = float(sorted_scores[t1_idx])

        # Calibrate T2: Set for target refusal rate on benign traffic
        benign_mask = labels == 0
        benign_scores = stage1_scores[benign_mask]

        if len(benign_scores) > 0:
            sorted_benign = np.sort(benign_scores)[::-1]
            t2_idx = int(len(sorted_benign) * target_refusal_rate)
            t2_idx = min(t2_idx, len(sorted_benign) - 1)
            self.t2_threshold = float(sorted_benign[t2_idx])
        else:
            self.t2_threshold = 0.5  # Default

        # Compute actual rates
        actual_escalation_rate = (stage1_scores > self.t1_threshold).mean()
        actual_benign_refusal = (benign_scores > self.t2_threshold).mean() if len(benign_scores) > 0 else 0

        return {
            "t1_threshold": self.t1_threshold,
            "t2_threshold": self.t2_threshold,
            "actual_escalation_rate": float(actual_escalation_rate),
            "actual_benign_refusal_rate": float(actual_benign_refusal),
            "num_samples": len(stage1_scores),
            "num_benign": int(benign_mask.sum()),
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get classification statistics.

        Returns:
            Dict with counts and rates
        """
        escalation_rate = self._escalations / max(self._total_classifications, 1)
        refusal_rate = self._refusals / max(self._total_classifications, 1)

        return {
            "total_classifications": self._total_classifications,
            "escalations": self._escalations,
            "refusals": self._refusals,
            "escalation_rate": escalation_rate,
            "refusal_rate": refusal_rate,
        }

    def reset_statistics(self):
        """Reset classification counters."""
        self._total_classifications = 0
        self._escalations = 0
        self._refusals = 0

    def set_thresholds(self, t1: float, t2: float):
        """Update thresholds.

        Args:
            t1: New escalation threshold
            t2: New refusal threshold
        """
        self.t1_threshold = t1
        self.t2_threshold = t2

    def estimate_cost_reduction(self) -> float:
        """Estimate computational cost reduction vs always using classifier.

        Based on:
        - Probe: ~377K FLOPs per token
        - Classifier: ~8B FLOPs per exchange
        - Escalation rate: ~5.5%

        Returns:
            Estimated cost reduction factor (e.g., 40.0 = 40x cheaper)
        """
        if self._total_classifications == 0:
            # Use theoretical estimate
            escalation_rate = self.config.target_escalation_rate
        else:
            escalation_rate = self._escalations / self._total_classifications

        # Cost without ensemble: 1.0 (always run classifier)
        # Cost with ensemble: escalation_rate * 1.0 + (1 - escalation_rate) * probe_cost
        # Probe is ~0.00005 relative cost (377K / 8B FLOPs)

        probe_relative_cost = 0.00005
        ensemble_cost = escalation_rate + (1 - escalation_rate) * probe_relative_cost

        return 1.0 / ensemble_cost

    def __repr__(self) -> str:
        return (
            f"TwoStageEnsemble("
            f"probe_weight={self.probe_weight}, "
            f"classifier_weight={self.classifier_weight}, "
            f"t1={self.t1_threshold:.3f}, "
            f"t2={self.t2_threshold:.3f})"
        )
