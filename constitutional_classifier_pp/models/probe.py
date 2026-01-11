"""Linear Activation Probe for Stage 1 classification.

This module implements the linear probe that monitors model activations
in real-time during generation, providing cheap initial screening for
potentially harmful content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config.base import ProbeConfig


@dataclass
class ProbeOutput:
    """Output from the linear activation probe."""

    logits: torch.Tensor  # Raw logits [batch, seq_len, 1] or [batch, 1]
    probabilities: torch.Tensor  # Sigmoid probabilities
    max_probability: float  # Max probability across sequence
    should_escalate: bool  # Whether to escalate to Stage 2


class LinearActivationProbe(nn.Module):
    """Stage 1: Linear probe on concatenated layer activations.

    Architecture: p_probe(y=1|x_{1:t}) = sigmoid(W^T * psi_t + b)
    where psi_t = [phi_t^(l1); phi_t^(l2); ...] concatenated across layers.

    The probe uses EMA smoothing for stable streaming inference and supports
    both training (full sequence) and inference (token-by-token) modes.

    Example:
        >>> config = ProbeConfig()
        >>> probe = LinearActivationProbe(config)
        >>> # Full sequence mode
        >>> logits = probe(activations)  # [batch, seq_len, 1]
        >>> # Streaming mode
        >>> probe.reset_ema()
        >>> for token_act in token_activations:
        ...     score = probe.forward_streaming(token_act)
    """

    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config

        # Core linear layer: W^T * psi + b
        self.linear = nn.Linear(config.probe_input_dim, config.output_dim, bias=True)

        # Initialize weights (Xavier for stability)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # EMA state for streaming inference (registered as buffer for device handling)
        self.register_buffer("ema_state", None)
        self.ema_alpha = config.ema_alpha

        # Threshold for escalation
        self.t1_threshold = config.t1_threshold

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Forward pass for full sequence processing.

        Args:
            activations: [batch, seq_len, hidden_dim * num_layers]
                Concatenated activations from all probed layers

        Returns:
            logits: [batch, seq_len, 1] - Pre-sigmoid logits
        """
        return self.linear(activations)

    def forward_with_ema(
        self,
        activations: torch.Tensor,
        reset_ema: bool = False
    ) -> torch.Tensor:
        """Forward pass with EMA smoothing for streaming inference.

        EMA formula: z_ema = alpha * z_current + (1 - alpha) * z_prev

        This provides stable scores that don't spike on individual tokens,
        making the probe more robust to isolated suspicious activations.

        Args:
            activations: [batch, 1, hidden_dim * num_layers] - Single token
            reset_ema: Reset EMA state for new sequence

        Returns:
            smoothed_logit: [batch, 1] - EMA-smoothed logit
        """
        batch_size = activations.shape[0]
        device = activations.device

        if reset_ema or self.ema_state is None:
            self.ema_state = torch.zeros(batch_size, 1, device=device)

        # Compute current logit
        current_logit = self.linear(activations).squeeze(-1)  # [batch, 1]

        # EMA update
        self.ema_state = (
            self.ema_alpha * current_logit +
            (1 - self.ema_alpha) * self.ema_state
        )

        return self.ema_state

    def forward_streaming(
        self,
        activations: torch.Tensor,
        reset: bool = False
    ) -> ProbeOutput:
        """Streaming inference for token-by-token generation.

        Args:
            activations: [batch, 1, hidden_dim * num_layers] or
                        [batch, hidden_dim * num_layers]
            reset: Reset EMA state for new sequence

        Returns:
            ProbeOutput with smoothed scores and escalation decision
        """
        # Handle 2D input
        if activations.dim() == 2:
            activations = activations.unsqueeze(1)

        logits = self.forward_with_ema(activations, reset_ema=reset)
        probs = torch.sigmoid(logits)
        max_prob = probs.max().item()

        return ProbeOutput(
            logits=logits,
            probabilities=probs,
            max_probability=max_prob,
            should_escalate=max_prob > self.t1_threshold,
        )

    def forward_with_window(
        self,
        activations: torch.Tensor,
        window_size: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass with sliding window averaging.

        Alternative to EMA: z_bar_t = (1/M) * sum_{k=0}^{M-1} z_{t-k}

        Args:
            activations: [batch, seq_len, hidden_dim * num_layers]
            window_size: Window size M (default from config)

        Returns:
            windowed_logits: [batch, seq_len, 1]
        """
        window_size = window_size or self.config.window_size

        # Get raw logits
        logits = self.linear(activations)  # [batch, seq_len, 1]
        logits = logits.squeeze(-1)  # [batch, seq_len]

        # Apply sliding window average using unfold
        batch_size, seq_len = logits.shape

        # Pad for valid convolution
        padded = F.pad(logits, (window_size - 1, 0), value=0.0)

        # Use unfold for sliding window
        windows = padded.unfold(1, window_size, 1)  # [batch, seq_len, window_size]
        z_bar = windows.mean(dim=2)  # [batch, seq_len]

        return z_bar.unsqueeze(-1)  # [batch, seq_len, 1]

    def get_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probability."""
        return torch.sigmoid(logits)

    def get_max_probability(self, logits: torch.Tensor) -> float:
        """Get maximum probability across sequence."""
        probs = self.get_probability(logits)
        return probs.max().item()

    def should_escalate(
        self,
        logits: torch.Tensor,
        threshold: Optional[float] = None
    ) -> bool:
        """Check if exchange should be escalated to Stage 2.

        Args:
            logits: Probe logits
            threshold: Override threshold (default: self.t1_threshold)

        Returns:
            True if max probability exceeds threshold
        """
        threshold = threshold if threshold is not None else self.t1_threshold
        max_prob = self.get_max_probability(logits)
        return max_prob > threshold

    def classify_exchange(
        self,
        activations: torch.Tensor,
        use_window: bool = False
    ) -> ProbeOutput:
        """Classify a complete exchange.

        Args:
            activations: [batch, seq_len, hidden_dim * num_layers]
            use_window: Use sliding window averaging

        Returns:
            ProbeOutput with classification results
        """
        if use_window:
            logits = self.forward_with_window(activations)
        else:
            logits = self.forward(activations)

        probs = self.get_probability(logits)
        max_prob = probs.max().item()

        return ProbeOutput(
            logits=logits,
            probabilities=probs,
            max_probability=max_prob,
            should_escalate=max_prob > self.t1_threshold,
        )

    def reset_ema(self):
        """Reset EMA state for new conversation/sequence."""
        self.ema_state = None

    def set_threshold(self, threshold: float):
        """Update escalation threshold."""
        self.t1_threshold = threshold

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs for single token inference.

        For a linear layer: 2 * input_dim * output_dim (multiply-add)
        """
        return 2 * self.config.probe_input_dim * self.config.output_dim

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[ProbeConfig] = None) -> "LinearActivationProbe":
        """Load a pretrained probe from disk.

        Args:
            path: Path to saved probe state dict
            config: Optional config (will be inferred from weights if not provided)

        Returns:
            Loaded LinearActivationProbe
        """
        state_dict = torch.load(path, map_location="cpu")

        if config is None:
            # Infer config from weight shapes
            weight_shape = state_dict["linear.weight"].shape
            config = ProbeConfig()
            # Update probe_input_dim based on weight shape
            input_dim = weight_shape[1]
            config.num_layers = input_dim // config.hidden_dim

        probe = cls(config)
        probe.load_state_dict(state_dict, strict=False)
        return probe

    def save_pretrained(self, path: str):
        """Save probe to disk.

        Args:
            path: Path to save probe state dict
        """
        torch.save(self.state_dict(), path)

    def __repr__(self) -> str:
        return (
            f"LinearActivationProbe("
            f"input_dim={self.config.probe_input_dim}, "
            f"output_dim={self.config.output_dim}, "
            f"ema_alpha={self.ema_alpha}, "
            f"threshold={self.t1_threshold})"
        )
