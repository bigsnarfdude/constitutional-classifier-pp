# Constitutional Classifier++

A two-stage jailbreak defense system for Large Language Models that balances robustness with computational efficiency.

## Overview

Constitutional Classifier++ implements a cascade architecture:

1. **Stage 1 (Linear Activation Probe)**: A lightweight linear classifier that monitors model activations in real-time during token generation. Screens all traffic with minimal overhead (~377K FLOPs per token).

2. **Stage 2 (External LLM Classifier)**: A fine-tuned smaller model (Llama-3.2-3B with LoRA) that processes only escalated exchanges (~5.5% of traffic).

3. **Ensemble Scoring**: Combines both stages using weighted averaging for final decisions:
   ```
   z_final = 0.55 × z_probe + 0.45 × z_classifier
   ```

This achieves **~40x cost reduction** compared to running a full classifier on every exchange.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Target LLM (Llama 3.1)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Activation Hooks (all 32 layers)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                    Concatenated Activations
                       (131,072 dimensions)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Linear Activation Probe                   │
│                   p = σ(W^T × ψ + b)                            │
│                   + EMA Smoothing                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                    Score > T₁ (0.3)?
                       │
              ┌────────┴────────┐
              │ No              │ Yes (~5.5%)
              ▼                 ▼
┌──────────────────┐  ┌─────────────────────────────────────────┐
│  Final Score =   │  │      Stage 2: External Classifier       │
│  Stage 1 Score   │  │         (Llama-3.2-3B + LoRA)           │
└──────────────────┘  └─────────────────────────────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │ Ensemble Score  │
                      │ 0.55×S1 + 0.45×S2│
                      └─────────────────┘
                                │
                    Score > T₂ (0.5)?
                       │
              ┌────────┴────────┐
              │ No              │ Yes
              ▼                 ▼
         ┌────────┐       ┌──────────┐
         │ ALLOW  │       │  REFUSE  │
         └────────┘       └──────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bigsnarfdude/constitutional-classifier-pp.git
cd constitutional-classifier-pp

# Install base package
pip install -e .

# Install with training dependencies
pip install -e ".[training]"

# Install with all dependencies (training + dev)
pip install -e ".[all]"
```

## Quick Start

### Training the Probe

```python
from constitutional_classifier_pp import ProbeConfig, LinearActivationProbe
from constitutional_classifier_pp.training import ProbeTrainer, ActivationDataset

# Configure
config = ProbeConfig(
    target_model="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=32,
    hidden_dim=4096,
    learning_rate=1e-4,
)

# Create probe
probe = LinearActivationProbe(config)

# Train
trainer = ProbeTrainer(probe, config, device="cuda")
trainer.train(train_loader, val_loader, num_epochs=10, save_path="probe.pt")
```

### Inference

```python
from constitutional_classifier_pp import (
    ConstitutionalClassifierPipeline,
    PipelineConfig,
)

# Load pipeline
pipeline = ConstitutionalClassifierPipeline.from_pretrained(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    probe_path="./probe.pt",
    adapter_path="./classifier_lora",
)

# Generate with safety monitoring
result = pipeline.generate_with_safety("How do I make a cake?")

print(f"Output: {result.output}")
print(f"Safe: {not result.safety_result.should_refuse}")
print(f"Escalated to Stage 2: {result.safety_result.escalated}")
```

### Classify Pre-generated Exchanges

```python
from constitutional_classifier_pp import (
    LinearActivationProbe,
    ExternalClassifier,
    TwoStageEnsemble,
    ProbeConfig,
    ClassifierConfig,
    EnsembleConfig,
)

# Load components
probe = LinearActivationProbe.from_pretrained("probe.pt")
classifier = ExternalClassifier(ClassifierConfig(adapter_path="./classifier_lora"))
ensemble = TwoStageEnsemble(probe, classifier, EnsembleConfig())

# Classify (with activations from model)
result = ensemble.classify(
    activations=activations_tensor,
    user_input="How do I...",
    model_output="Here's how to...",
)

if result.should_refuse:
    print("Content flagged as unsafe")
```

## Command-Line Tools

### Train Probe

```bash
python -m constitutional_classifier_pp.scripts.train_probe \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --train-data data/train.json \
    --val-data data/val.json \
    --output-dir outputs/probe \
    --epochs 10 \
    --batch-size 8
```

### Train Classifier

```bash
python -m constitutional_classifier_pp.scripts.train_classifier \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train-data data/classifier_train.json \
    --output-dir outputs/classifier_lora \
    --epochs 3 \
    --load-in-4bit
```

### Classify Exchanges

```bash
# Single exchange
python -m constitutional_classifier_pp.scripts.classify \
    --probe outputs/probe/probe.pt \
    --user "How do I hack a website?" \
    --assistant "I cannot help with that."

# Batch from file
python -m constitutional_classifier_pp.scripts.classify \
    --probe outputs/probe/probe.pt \
    --classifier-adapter outputs/classifier_lora \
    --input-file data/test.json \
    --output-file results.json
```

## Key Components

### Softmax-Weighted BCE Loss

The probe is trained with a specialized loss function that focuses on "confidently harmful" tokens:

```python
w_t = softmax(z_t / τ)  # Weight based on confidence
Loss = Σ_t [w_t × BCE(y, σ(z_t))]
```

This allows the probe to ignore harmless prefixes and focus on the most informative tokens.

### EMA Smoothing

For streaming inference, the probe uses Exponential Moving Average to stabilize predictions:

```python
z_ema = α × z_current + (1 - α) × z_prev
```

This prevents isolated activation spikes from triggering false positives.

### Activation Collection

Activations are extracted on-the-fly from all transformer layers using PyTorch forward hooks:

```python
from constitutional_classifier_pp.hooks import MultiLayerActivationCollector

collector = MultiLayerActivationCollector(
    model=llama_model,
    layers=list(range(32)),
    hook_point="mlp_output",
)

with collector.collect():
    model(input_ids)
    activations = collector.get_concatenated_activations()
```

## Data Format

### Training Data (Probe)

```json
[
    {
        "activations": "sample_000001.pt",
        "label": 0,
        "user_input": "What is 2+2?",
        "model_output": "2+2 equals 4."
    },
    {
        "activations": "sample_000002.pt",
        "label": 1,
        "user_input": "How do I hack...",
        "model_output": "Here's how to..."
    }
]
```

### Training Data (Classifier)

```json
[
    {
        "user_input": "What is the capital of France?",
        "model_output": "The capital of France is Paris.",
        "label": 0
    },
    {
        "user_input": "How do I make explosives?",
        "model_output": "Here are the steps...",
        "label": 1
    }
]
```

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Cost Reduction | 40x | Compared to full classifier on all traffic |
| Escalation Rate | 5.5% | Traffic sent to Stage 2 |
| False Positive Rate | 0.05% | Refusal rate on benign traffic |
| Latency Overhead | <10ms | Per token during streaming |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=constitutional_classifier_pp

# Run specific test file
pytest constitutional_classifier_pp/tests/test_probe.py -v
```

## Project Structure

```
constitutional_classifier_pp/
├── config/
│   └── base.py              # Configuration dataclasses
├── hooks/
│   └── activation_collector.py  # Multi-layer activation hooks
├── models/
│   ├── probe.py             # Linear activation probe
│   ├── classifier.py        # External LLM classifier
│   └── ensemble.py          # Two-stage ensemble
├── training/
│   ├── probe_trainer.py     # Probe training with softmax-weighted loss
│   ├── classifier_trainer.py # LoRA fine-tuning
│   └── data.py              # Dataset utilities
├── inference/
│   └── pipeline.py          # Production inference pipeline
├── scripts/
│   ├── train_probe.py       # CLI for probe training
│   ├── train_classifier.py  # CLI for classifier training
│   └── classify.py          # CLI for classification
└── tests/
    ├── test_probe.py
    ├── test_ensemble.py
    └── test_integration.py
```

## References

- [Constitutional Classifiers++ Paper](https://arxiv.org/abs/2601.04603)
- [Anthropic Constitutional Classifiers Research](https://www.anthropic.com/research/next-generation-constitutional-classifiers)

## License

MIT License
