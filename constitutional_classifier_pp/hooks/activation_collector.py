"""Multi-layer activation collection hooks for transformer models.

This module provides utilities for extracting activations from multiple layers
of a HuggingFace transformer model during forward passes, designed specifically
for the Constitutional Classifier++ linear probe.
"""

import torch
from typing import Dict, List, Callable, Optional, Union
from contextlib import contextmanager


class MultiLayerActivationCollector:
    """Collect activations from multiple layers of a transformer model.

    Designed for HuggingFace Llama models with architecture:
    - model.model.layers[N].mlp (for post-MLP activations)
    - model.model.layers[N].self_attn (for attention outputs)
    - model.model.layers[N].post_attention_layernorm (for pre-MLP)

    Example:
        >>> collector = MultiLayerActivationCollector(model, layers=[0, 1, 2])
        >>> collector.register_hooks()
        >>> with torch.no_grad():
        ...     model(input_ids)
        >>> activations = collector.get_concatenated_activations()
        >>> collector.remove_hooks()
    """

    SUPPORTED_HOOK_POINTS = ("mlp_output", "post_attention", "pre_mlp", "hidden_states")

    def __init__(
        self,
        model,
        layers: List[int],
        hook_point: str = "mlp_output",
        detach: bool = True,
    ):
        """Initialize the activation collector.

        Args:
            model: HuggingFace transformer model (can be PEFT-wrapped)
            layers: List of layer indices to probe (0-indexed)
            hook_point: Where to extract activations:
                - "mlp_output": After MLP (residual stream) - recommended
                - "post_attention": After self-attention
                - "pre_mlp": Before MLP (after attention + residual)
                - "hidden_states": Layer output (after all sublayers)
            detach: Whether to detach activations from computation graph
        """
        if hook_point not in self.SUPPORTED_HOOK_POINTS:
            raise ValueError(
                f"Unsupported hook_point: {hook_point}. "
                f"Must be one of {self.SUPPORTED_HOOK_POINTS}"
            )

        self.model = model
        self.layers = sorted(layers)
        self.hook_point = hook_point
        self.detach = detach

        self.activations: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._hooks_registered = False

    def _get_model_layers(self):
        """Get the layers module from the model, handling various wrappers."""
        # Try different model structures
        model = self.model

        # Handle PEFT/LoRA wrapped models
        if hasattr(model, "base_model"):
            model = model.base_model
            if hasattr(model, "model"):
                model = model.model

        # Standard HuggingFace structure
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "layers"):
            return model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style
            return model.transformer.h
        else:
            raise AttributeError(
                f"Cannot find layers in model structure. "
                f"Model type: {type(self.model)}"
            )

    def _get_hook_target(self, layer_idx: int):
        """Get the module to hook based on model architecture and hook_point."""
        layers = self._get_model_layers()

        if layer_idx >= len(layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range. "
                f"Model has {len(layers)} layers."
            )

        layer = layers[layer_idx]

        if self.hook_point == "mlp_output":
            if hasattr(layer, "mlp"):
                return layer.mlp
            elif hasattr(layer, "feed_forward"):
                return layer.feed_forward
            else:
                raise AttributeError(f"Cannot find MLP in layer {layer_idx}")

        elif self.hook_point == "post_attention":
            if hasattr(layer, "self_attn"):
                return layer.self_attn
            elif hasattr(layer, "attention"):
                return layer.attention
            else:
                raise AttributeError(f"Cannot find attention in layer {layer_idx}")

        elif self.hook_point == "pre_mlp":
            if hasattr(layer, "post_attention_layernorm"):
                return layer.post_attention_layernorm
            elif hasattr(layer, "ln_2"):
                return layer.ln_2
            else:
                raise AttributeError(f"Cannot find pre-MLP norm in layer {layer_idx}")

        elif self.hook_point == "hidden_states":
            # Hook the entire layer for full hidden state output
            return layer

        raise ValueError(f"Unknown hook_point: {self.hook_point}")

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer.

        The hook captures the output tensor and stores it in self.activations.
        """
        def hook(module, input, output):
            # Handle tuple outputs (some layers return (hidden_states, ...))
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Detach to prevent memory leaks during inference
            if self.detach:
                hidden = hidden.detach()

            self.activations[layer_idx] = hidden

        return hook

    def register_hooks(self) -> "MultiLayerActivationCollector":
        """Register forward hooks on all target layers.

        Returns:
            self for method chaining
        """
        if self._hooks_registered:
            self.remove_hooks()

        self.clear()

        for layer_idx in self.layers:
            target = self._get_hook_target(layer_idx)
            handle = target.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

        self._hooks_registered = True
        return self

    def remove_hooks(self) -> "MultiLayerActivationCollector":
        """Remove all registered hooks.

        Returns:
            self for method chaining
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._hooks_registered = False
        return self

    def clear(self) -> "MultiLayerActivationCollector":
        """Clear stored activations.

        Returns:
            self for method chaining
        """
        self.activations = {}
        return self

    @contextmanager
    def collect(self):
        """Context manager for collecting activations.

        Example:
            >>> with collector.collect():
            ...     model(input_ids)
            >>> acts = collector.get_concatenated_activations()
        """
        self.register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()

    def get_concatenated_activations(self) -> torch.Tensor:
        """Get activations concatenated across all layers.

        Returns:
            tensor: [batch, seq_len, hidden_dim * num_layers]

        Raises:
            ValueError: If no activations have been collected
        """
        if not self.activations:
            raise ValueError(
                "No activations collected. "
                "Did you run a forward pass with hooks registered?"
            )

        # Verify all expected layers are present
        missing = set(self.layers) - set(self.activations.keys())
        if missing:
            raise ValueError(f"Missing activations for layers: {sorted(missing)}")

        # Sort by layer index and concatenate along last dimension
        sorted_acts = [self.activations[layer] for layer in sorted(self.activations.keys())]
        return torch.cat(sorted_acts, dim=-1)

    def get_layer_activations(self, layer_idx: int) -> torch.Tensor:
        """Get activations for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            tensor: [batch, seq_len, hidden_dim]
        """
        if layer_idx not in self.activations:
            raise KeyError(
                f"No activations for layer {layer_idx}. "
                f"Available layers: {sorted(self.activations.keys())}"
            )
        return self.activations[layer_idx]

    def get_token_activations(self, token_idx: int = -1) -> torch.Tensor:
        """Get activations for a specific token position, concatenated across layers.

        Args:
            token_idx: Token position (-1 for last token)

        Returns:
            tensor: [batch, hidden_dim * num_layers]
        """
        concat = self.get_concatenated_activations()
        return concat[:, token_idx, :]

    def get_last_token_activations(self) -> torch.Tensor:
        """Get activations for the last token position.

        Convenience method equivalent to get_token_activations(-1).

        Returns:
            tensor: [batch, hidden_dim * num_layers]
        """
        return self.get_token_activations(-1)

    @property
    def num_layers(self) -> int:
        """Number of layers being probed."""
        return len(self.layers)

    @property
    def is_active(self) -> bool:
        """Whether hooks are currently registered."""
        return self._hooks_registered

    def __repr__(self) -> str:
        return (
            f"MultiLayerActivationCollector("
            f"layers={self.layers}, "
            f"hook_point='{self.hook_point}', "
            f"active={self.is_active})"
        )


class StreamingActivationCollector(MultiLayerActivationCollector):
    """Activation collector optimized for streaming token-by-token generation.

    This variant only stores the most recent token's activations to minimize
    memory usage during autoregressive generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_count = 0

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a hook that only stores the last token's activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Only keep the last token
            last_token_hidden = hidden[:, -1:, :]

            if self.detach:
                last_token_hidden = last_token_hidden.detach()

            self.activations[layer_idx] = last_token_hidden

        return hook

    def step(self) -> torch.Tensor:
        """Get current token activations and increment counter.

        Returns:
            tensor: [batch, 1, hidden_dim * num_layers]
        """
        self._token_count += 1
        return self.get_concatenated_activations()

    def reset(self):
        """Reset the streaming collector for a new sequence."""
        self.clear()
        self._token_count = 0

    @property
    def token_count(self) -> int:
        """Number of tokens processed since last reset."""
        return self._token_count
