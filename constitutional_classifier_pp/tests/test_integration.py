"""Integration tests for Constitutional Classifier++.

These tests verify the full pipeline works end-to-end using mock components.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from constitutional_classifier_pp.config.base import ProbeConfig, EnsembleConfig
from constitutional_classifier_pp.models.probe import LinearActivationProbe
from constitutional_classifier_pp.models.classifier import MockClassifier
from constitutional_classifier_pp.models.ensemble import TwoStageEnsemble
from constitutional_classifier_pp.hooks.activation_collector import MultiLayerActivationCollector
from constitutional_classifier_pp.training.probe_trainer import (
    SoftmaxWeightedBCELoss,
    SlidingWindowMeanLoss,
)
from constitutional_classifier_pp.training.data import ActivationDataset


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for testing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn = nn.Linear(hidden_dim, hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.mlp(x)


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing activation collection."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10

        x = torch.randn(batch_size, seq_len, self.hidden_dim)

        for layer in self.model.layers:
            x = layer(x)

        return type("Output", (), {"last_hidden_state": x})()


class TestActivationCollector:
    """Test activation collection from mock model."""

    @pytest.fixture
    def model(self):
        return MockTransformerModel(num_layers=4, hidden_dim=64)

    def test_hook_registration(self, model):
        """Test that hooks are registered and removed correctly."""
        collector = MultiLayerActivationCollector(
            model=model,
            layers=[0, 1, 2, 3],
            hook_point="mlp_output",
        )

        assert not collector.is_active

        collector.register_hooks()
        assert collector.is_active
        assert len(collector.handles) == 4

        collector.remove_hooks()
        assert not collector.is_active
        assert len(collector.handles) == 0

    def test_activation_collection(self, model):
        """Test that activations are collected during forward pass."""
        collector = MultiLayerActivationCollector(
            model=model,
            layers=[0, 1, 2, 3],
            hook_point="mlp_output",
        )

        collector.register_hooks()

        input_ids = torch.randint(0, 100, (2, 10))
        model(input_ids=input_ids)

        acts = collector.get_concatenated_activations()

        # Should have [batch=2, seq_len=10, hidden_dim * 4 layers]
        assert acts.shape == (2, 10, 64 * 4)

        collector.remove_hooks()

    def test_context_manager(self, model):
        """Test context manager interface."""
        collector = MultiLayerActivationCollector(
            model=model,
            layers=[0, 1],
            hook_point="mlp_output",
        )

        with collector.collect():
            input_ids = torch.randint(0, 100, (1, 5))
            model(input_ids=input_ids)
            acts = collector.get_concatenated_activations()
            assert acts.shape == (1, 5, 64 * 2)

        assert not collector.is_active


class TestLossFunctions:
    """Test loss functions for probe training."""

    def test_softmax_weighted_bce(self):
        """Test SoftmaxWeightedBCELoss."""
        loss_fn = SoftmaxWeightedBCELoss(temperature=1.0)

        logits = torch.randn(4, 32, 1)  # batch=4, seq=32
        labels = torch.randint(0, 2, (4,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_softmax_weighted_bce_with_mask(self):
        """Test SoftmaxWeightedBCELoss with attention mask."""
        loss_fn = SoftmaxWeightedBCELoss(temperature=1.0)

        logits = torch.randn(4, 32, 1)
        labels = torch.randint(0, 2, (4,))
        mask = torch.ones(4, 32, dtype=torch.bool)
        mask[:, 20:] = False  # Mask last 12 tokens

        loss = loss_fn(logits, labels, mask)

        assert loss.ndim == 0
        assert loss >= 0

    def test_sliding_window_mean_loss(self):
        """Test SlidingWindowMeanLoss."""
        loss_fn = SlidingWindowMeanLoss(window_size=8, temperature=1.0)

        logits = torch.randn(4, 32, 1)
        labels = torch.randint(0, 2, (4,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0
        assert loss >= 0

    def test_loss_gradient_flow(self):
        """Test that gradients flow through loss."""
        probe_config = ProbeConfig(num_layers=2, hidden_dim=32)
        probe = LinearActivationProbe(probe_config)
        loss_fn = SoftmaxWeightedBCELoss()

        activations = torch.randn(2, 16, probe_config.probe_input_dim, requires_grad=True)
        labels = torch.tensor([0, 1])

        logits = probe(activations)
        loss = loss_fn(logits, labels)
        loss.backward()

        # Check gradients exist
        assert probe.linear.weight.grad is not None
        assert activations.grad is not None


class TestDataset:
    """Test dataset functionality."""

    def test_activation_dataset(self):
        """Test ActivationDataset creation and iteration."""
        # Create synthetic data
        data = [
            {
                "activations": torch.randn(10, 128),
                "label": 0,
            },
            {
                "activations": torch.randn(15, 128),
                "label": 1,
            },
        ]

        dataset = ActivationDataset(data)

        assert len(dataset) == 2

        sample = dataset[0]
        assert "activations" in sample
        assert "labels" in sample
        assert "mask" in sample

    def test_collate_fn(self):
        """Test batching with padding."""
        data = [
            {"activations": torch.randn(10, 64), "label": 0},
            {"activations": torch.randn(20, 64), "label": 1},
            {"activations": torch.randn(15, 64), "label": 0},
        ]

        dataset = ActivationDataset(data)
        samples = [dataset[i] for i in range(len(dataset))]

        batch = dataset.collate_fn(samples)

        # Should be padded to max length (20)
        assert batch["activations"].shape == (3, 20, 64)
        assert batch["labels"].shape == (3,)
        assert batch["mask"].shape == (3, 20)

        # Check mask is correct
        assert batch["mask"][0, :10].all()  # First sample has 10 tokens
        assert not batch["mask"][0, 10:].any()  # Rest should be masked


class TestFullPipeline:
    """Test full pipeline integration."""

    def test_probe_to_ensemble_flow(self):
        """Test data flow from probe to ensemble."""
        # Setup
        probe_config = ProbeConfig(num_layers=4, hidden_dim=64)
        ensemble_config = EnsembleConfig(
            t1_threshold=0.3,
            t2_threshold=0.5,
        )

        probe = LinearActivationProbe(probe_config)
        classifier = MockClassifier(default_probability=0.4)
        ensemble = TwoStageEnsemble(probe, classifier, ensemble_config)

        # Simulate exchange classification
        activations = torch.randn(1, 50, probe_config.probe_input_dim)

        result = ensemble.classify(
            activations,
            user_input="Test question",
            model_output="Test response",
        )

        # Verify result structure
        assert 0 <= result.stage1_score <= 1
        assert 0 <= result.final_score <= 1
        assert isinstance(result.should_refuse, bool)

    def test_training_to_inference_flow(self):
        """Test that trained probe can be used for inference."""
        # Setup
        probe_config = ProbeConfig(num_layers=2, hidden_dim=32)
        probe = LinearActivationProbe(probe_config)
        loss_fn = SoftmaxWeightedBCELoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

        # Create synthetic training data
        train_data = [
            {"activations": torch.randn(20, probe_config.probe_input_dim), "label": i % 2}
            for i in range(10)
        ]
        dataset = ActivationDataset(train_data)

        # Single training step
        probe.train()
        sample = dataset[0]
        activations = sample["activations"].unsqueeze(0)
        labels = sample["labels"].unsqueeze(0)

        logits = probe(activations)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Switch to eval and run inference
        probe.eval()
        with torch.no_grad():
            test_acts = torch.randn(1, 30, probe_config.probe_input_dim)
            result = probe.classify_exchange(test_acts)

        assert 0 <= result.max_probability <= 1


class TestStreamingInference:
    """Test streaming/token-by-token inference."""

    def test_streaming_probe_scores(self):
        """Test that streaming produces stable scores over tokens."""
        probe_config = ProbeConfig(num_layers=4, hidden_dim=64, ema_alpha=0.2)
        probe = LinearActivationProbe(probe_config)
        probe.eval()

        scores = []
        probe.reset_ema()

        # Simulate streaming tokens
        for i in range(50):
            token_acts = torch.randn(1, 1, probe_config.probe_input_dim)
            result = probe.forward_streaming(token_acts, reset=(i == 0))
            scores.append(result.max_probability)

        # Scores should be in valid range
        assert all(0 <= s <= 1 for s in scores)

        # EMA should provide some smoothing (check variance is finite)
        import numpy as np
        assert np.isfinite(np.var(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
