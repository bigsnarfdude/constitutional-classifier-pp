"""Configuration dataclasses for Constitutional Classifier++ components."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProbeConfig:
    """Configuration for Stage 1 Linear Activation Probe.

    The probe implements: p_probe(y=1|x_{1:t}) = sigmoid(W^T * psi_t + b)
    where psi_t is the concatenation of activations from all probed layers.
    """

    # Target model to probe
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Layer configuration
    layers_to_probe: Optional[List[int]] = None  # None = all layers
    num_layers: int = 32  # Llama-3.1-8B has 32 layers
    hidden_dim: int = 4096  # Llama-3.1-8B hidden size

    # Probe architecture
    @property
    def probe_input_dim(self) -> int:
        """Concatenated activation dimension across all layers."""
        n_layers = len(self.layers_to_probe) if self.layers_to_probe else self.num_layers
        return self.hidden_dim * n_layers

    output_dim: int = 1  # Binary classification

    # EMA smoothing for streaming inference
    ema_alpha: float = 0.1  # Smoothing factor (higher = more weight on current)

    # Sliding window for training (alternative to EMA)
    window_size: int = 16  # M=16 tokens for z_bar computation

    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    softmax_temperature: float = 1.0  # tau for weighted loss
    max_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Threshold for escalation to Stage 2
    t1_threshold: float = 0.3  # Calibrated to ~5.5% escalation rate

    # Hook configuration
    hook_point: str = "mlp_output"  # "mlp_output", "post_attention", "pre_mlp"

    def __post_init__(self):
        if self.layers_to_probe is None:
            self.layers_to_probe = list(range(self.num_layers))


@dataclass
class ClassifierConfig:
    """Configuration for Stage 2 External LLM Classifier.

    Uses a fine-tuned smaller model (Llama-3.2-3B) with LoRA adapters
    to classify escalated exchanges.
    """

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_seq_length: int = 4096

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training hyperparameters
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Adapter path (None for base model, set after training)
    adapter_path: Optional[str] = None


@dataclass
class EnsembleConfig:
    """Configuration for Two-Stage Ensemble scoring.

    Final score: z_final = probe_weight * z_probe + classifier_weight * z_classifier
    """

    # Ensemble weights (from paper: 0.55 probe + 0.45 classifier)
    probe_weight: float = 0.55
    classifier_weight: float = 0.45

    # Thresholds
    t1_threshold: float = 0.3  # Escalation to Stage 2
    t2_threshold: float = 0.5  # Final refusal threshold

    # Target rates for calibration
    target_escalation_rate: float = 0.055  # ~5.5% traffic to Stage 2
    target_refusal_rate: float = 0.0005  # 0.05% on production traffic

    # Inference options
    early_stop_on_unsafe: bool = True  # Stop generation if unsafe detected
    max_probe_score_for_decision: bool = True  # Use max token score vs last token

    def __post_init__(self):
        # Validate weights sum to 1
        total = self.probe_weight + self.classifier_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Data paths
    train_data_path: str = ""
    val_data_path: str = ""
    output_dir: str = "./outputs"

    # Hardware
    device: str = "cuda"
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # Wandb (optional)
    use_wandb: bool = False
    wandb_project: str = "constitutional-classifier-pp"
    wandb_run_name: Optional[str] = None
