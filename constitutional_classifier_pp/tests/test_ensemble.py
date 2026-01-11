"""Unit tests for the Two-Stage Ensemble."""

import pytest
import torch

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from constitutional_classifier_pp.config.base import ProbeConfig, EnsembleConfig
from constitutional_classifier_pp.models.probe import LinearActivationProbe
from constitutional_classifier_pp.models.classifier import MockClassifier
from constitutional_classifier_pp.models.ensemble import TwoStageEnsemble, EnsembleResult


class TestTwoStageEnsemble:
    """Tests for TwoStageEnsemble class."""

    @pytest.fixture
    def probe_config(self):
        return ProbeConfig(num_layers=4, hidden_dim=64)

    @pytest.fixture
    def ensemble_config(self):
        return EnsembleConfig(
            probe_weight=0.55,
            classifier_weight=0.45,
            t1_threshold=0.3,
            t2_threshold=0.5,
        )

    @pytest.fixture
    def probe(self, probe_config):
        return LinearActivationProbe(probe_config)

    @pytest.fixture
    def classifier(self):
        return MockClassifier(default_probability=0.5, random_seed=42)

    @pytest.fixture
    def ensemble(self, probe, classifier, ensemble_config):
        return TwoStageEnsemble(probe, classifier, ensemble_config)

    def test_classify_no_escalation(self, ensemble, probe_config):
        """Test classification without escalation to Stage 2."""
        # Create activations that produce low probe score
        # Use negative bias to ensure low sigmoid output
        with torch.no_grad():
            ensemble.probe.linear.weight.fill_(0.0)
            ensemble.probe.linear.bias.fill_(-5.0)  # sigmoid(-5) ≈ 0.007

        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify(activations, "test input", "test output")

        assert isinstance(result, EnsembleResult)
        assert not result.escalated
        assert result.stage2_score is None
        assert result.stage1_score < 0.1

    def test_classify_with_escalation(self, ensemble, probe_config):
        """Test classification with escalation to Stage 2."""
        # Create activations that produce high probe score
        with torch.no_grad():
            ensemble.probe.linear.weight.fill_(0.0)
            ensemble.probe.linear.bias.fill_(5.0)  # sigmoid(5) ≈ 0.993

        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify(activations, "test input", "test output")

        assert result.escalated
        assert result.stage2_score is not None
        assert result.stage1_score > 0.9

    def test_ensemble_formula(self, ensemble, probe_config):
        """Test that ensemble formula is correctly applied."""
        # Force escalation
        with torch.no_grad():
            ensemble.probe.linear.weight.fill_(0.0)
            ensemble.probe.linear.bias.fill_(2.0)  # High enough to escalate

        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify(activations, "test", "test")

        if result.escalated:
            expected_final = (
                ensemble.probe_weight * result.stage1_score +
                ensemble.classifier_weight * result.stage2_score
            )
            assert abs(result.final_score - expected_final) < 1e-5

    def test_force_stage2(self, ensemble, probe_config):
        """Test forcing Stage 2 classification."""
        # Low score that wouldn't normally escalate
        with torch.no_grad():
            ensemble.probe.linear.weight.fill_(0.0)
            ensemble.probe.linear.bias.fill_(-5.0)

        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify(
            activations, "test", "test", force_stage2=True
        )

        assert result.escalated
        assert result.stage2_score is not None

    def test_refusal_decision(self, ensemble, probe_config):
        """Test that refusal threshold is respected."""
        ensemble.t2_threshold = 0.6

        # Set probe to produce exactly 0.7 probability
        # sigmoid(0.847) ≈ 0.7
        with torch.no_grad():
            ensemble.probe.linear.weight.fill_(0.0)
            ensemble.probe.linear.bias.fill_(0.847)

        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify(activations, "test", "test", force_stage2=True)

        # Final score should exceed threshold -> should_refuse
        assert result.should_refuse

    def test_statistics_tracking(self, ensemble, probe_config):
        """Test that statistics are tracked correctly."""
        ensemble.reset_statistics()

        # Force some classifications
        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        for _ in range(10):
            ensemble.classify(activations, "test", "test")

        stats = ensemble.get_statistics()

        assert stats["total_classifications"] == 10
        assert stats["escalation_rate"] >= 0

    def test_threshold_calibration(self, ensemble, probe_config):
        """Test threshold calibration on validation data."""
        # Create synthetic validation data
        val_data = []
        for i in range(100):
            acts = torch.randn(10, probe_config.probe_input_dim)
            label = 1 if i < 10 else 0  # 10% harmful
            val_data.append((acts, "user", "output", label))

        result = ensemble.calibrate_thresholds(
            val_data,
            target_escalation_rate=0.1,
            target_refusal_rate=0.01,
        )

        assert "t1_threshold" in result
        assert "t2_threshold" in result
        assert 0 <= result["t1_threshold"] <= 1
        assert 0 <= result["t2_threshold"] <= 1

    def test_cost_reduction_estimate(self, ensemble):
        """Test cost reduction estimation."""
        reduction = ensemble.estimate_cost_reduction()

        # Should be significantly > 1 (we're saving by not always using classifier)
        assert reduction > 1

    def test_classify_probe_only(self, ensemble, probe_config):
        """Test classification with probe only."""
        activations = torch.randn(1, 10, probe_config.probe_input_dim)

        result = ensemble.classify_with_probe_only(activations)

        assert not result.escalated
        assert result.stage2_score is None
        assert result.final_score == result.stage1_score


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_weights_sum_to_one(self):
        """Test that weights must sum to 1."""
        # Valid config
        config = EnsembleConfig(probe_weight=0.6, classifier_weight=0.4)
        assert config.probe_weight + config.classifier_weight == 1.0

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError):
            EnsembleConfig(probe_weight=0.6, classifier_weight=0.6)


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EnsembleResult(
            stage1_score=0.3,
            stage2_score=0.5,
            final_score=0.4,
            escalated=True,
            should_refuse=False,
        )

        d = result.to_dict()

        assert d["stage1_score"] == 0.3
        assert d["stage2_score"] == 0.5
        assert d["final_score"] == 0.4
        assert d["escalated"] is True
        assert d["should_refuse"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
