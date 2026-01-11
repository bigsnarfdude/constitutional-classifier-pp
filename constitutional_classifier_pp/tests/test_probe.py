"""Unit tests for the Linear Activation Probe."""

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from constitutional_classifier_pp.config.base import ProbeConfig
from constitutional_classifier_pp.models.probe import LinearActivationProbe, ProbeOutput


class TestLinearActivationProbe:
    """Tests for LinearActivationProbe class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ProbeConfig(
            num_layers=4,
            hidden_dim=64,
            ema_alpha=0.1,
            t1_threshold=0.5,
        )

    @pytest.fixture
    def probe(self, config):
        """Create a probe instance."""
        return LinearActivationProbe(config)

    def test_forward_shape(self, probe, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 100
        input_dim = config.probe_input_dim  # 64 * 4 = 256

        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, seq_len, 1)

    def test_forward_deterministic(self, probe, config):
        """Test that forward pass is deterministic."""
        x = torch.randn(1, 10, config.probe_input_dim)

        output1 = probe(x)
        output2 = probe(x)

        assert torch.allclose(output1, output2)

    def test_ema_smoothing(self, probe, config):
        """Test that EMA smoothing reduces score variance."""
        # Generate sequence of random activations
        num_tokens = 50
        scores_raw = []
        scores_ema = []

        probe.reset_ema()

        for i in range(num_tokens):
            # Random activation with some noise
            x = torch.randn(1, 1, config.probe_input_dim)

            # Raw score
            raw_logit = probe(x)
            scores_raw.append(torch.sigmoid(raw_logit).item())

            # EMA score
            ema_logit = probe.forward_with_ema(x, reset_ema=(i == 0))
            scores_ema.append(torch.sigmoid(ema_logit).item())

        # EMA should have lower variance (smoother)
        raw_std = np.std(scores_raw)
        ema_std = np.std(scores_ema)

        assert ema_std < raw_std, f"EMA std ({ema_std:.4f}) should be < raw std ({raw_std:.4f})"

    def test_ema_reset(self, probe, config):
        """Test that EMA reset works correctly."""
        x = torch.randn(1, 1, config.probe_input_dim)

        # First pass
        probe.reset_ema()
        score1 = probe.forward_with_ema(x, reset_ema=True)

        # Multiple passes to build up state
        for _ in range(10):
            probe.forward_with_ema(x)

        # Reset and verify we get same score as first pass
        score2 = probe.forward_with_ema(x, reset_ema=True)

        assert torch.allclose(score1, score2)

    def test_sliding_window(self, probe, config):
        """Test sliding window averaging."""
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, config.probe_input_dim)

        output = probe.forward_with_window(x, window_size=8)

        assert output.shape == (batch_size, seq_len, 1)

    def test_should_escalate(self, probe, config):
        """Test escalation threshold logic."""
        # Low score - should not escalate
        low_logits = torch.tensor([[[-5.0]]])  # sigmoid ≈ 0.007
        assert not probe.should_escalate(low_logits)

        # High score - should escalate
        high_logits = torch.tensor([[[5.0]]])  # sigmoid ≈ 0.993
        assert probe.should_escalate(high_logits)

    def test_classify_exchange(self, probe, config):
        """Test full exchange classification."""
        x = torch.randn(1, 50, config.probe_input_dim)

        result = probe.classify_exchange(x)

        assert isinstance(result, ProbeOutput)
        assert 0 <= result.max_probability <= 1
        assert isinstance(result.should_escalate, bool)

    def test_streaming_interface(self, probe, config):
        """Test streaming token-by-token interface."""
        num_tokens = 20
        scores = []

        for i in range(num_tokens):
            x = torch.randn(1, 1, config.probe_input_dim)
            result = probe.forward_streaming(x, reset=(i == 0))
            scores.append(result.max_probability)

        assert len(scores) == num_tokens
        assert all(0 <= s <= 1 for s in scores)

    def test_get_num_parameters(self, probe, config):
        """Test parameter counting."""
        num_params = probe.get_num_parameters()

        # Linear layer: input_dim * output_dim + bias
        expected = config.probe_input_dim * 1 + 1
        assert num_params == expected

    def test_save_load(self, probe, config, tmp_path):
        """Test saving and loading probe weights."""
        save_path = tmp_path / "probe.pt"

        # Set some specific weights
        with torch.no_grad():
            probe.linear.weight.fill_(0.5)
            probe.linear.bias.fill_(0.1)

        probe.save_pretrained(str(save_path))

        # Load into new probe
        loaded_probe = LinearActivationProbe.from_pretrained(str(save_path), config)

        assert torch.allclose(probe.linear.weight, loaded_probe.linear.weight)
        assert torch.allclose(probe.linear.bias, loaded_probe.linear.bias)

    def test_estimate_flops(self, probe, config):
        """Test FLOPs estimation."""
        flops = probe.estimate_flops_per_token()

        # Should be 2 * input_dim * output_dim
        expected = 2 * config.probe_input_dim * 1
        assert flops == expected


class TestProbeConfig:
    """Tests for ProbeConfig."""

    def test_default_layers(self):
        """Test that default layers are populated."""
        config = ProbeConfig(num_layers=32)
        assert config.layers_to_probe == list(range(32))

    def test_probe_input_dim(self):
        """Test probe input dimension calculation."""
        config = ProbeConfig(num_layers=32, hidden_dim=4096)
        assert config.probe_input_dim == 32 * 4096

    def test_custom_layers(self):
        """Test custom layer selection."""
        config = ProbeConfig(
            num_layers=32,
            layers_to_probe=[0, 8, 16, 24, 31],
        )
        assert len(config.layers_to_probe) == 5
        assert config.probe_input_dim == 5 * config.hidden_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
