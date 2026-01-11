#!/usr/bin/env python3
"""Train the Stage 1 Linear Activation Probe.

Example usage:
    python -m constitutional_classifier_pp.scripts.train_probe \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --train-data data/train.json \
        --val-data data/val.json \
        --output-dir outputs/probe \
        --epochs 10
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from constitutional_classifier_pp.config.base import ProbeConfig, TrainingConfig
from constitutional_classifier_pp.models.probe import LinearActivationProbe
from constitutional_classifier_pp.training.probe_trainer import ProbeTrainer
from constitutional_classifier_pp.training.data import ActivationDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Constitutional Classifier++ Stage 1 Probe"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Target model to probe",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of layers in target model",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension of target model",
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data JSON",
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        default=None,
        help="Directory containing activation files",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for weighted loss",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["softmax_weighted", "sliding_window"],
        default="softmax_weighted",
        help="Loss function type",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/probe",
        help="Output directory for trained probe",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Constitutional Classifier++ Probe Training")
    print("=" * 60)

    # Create config
    probe_config = ProbeConfig(
        target_model=args.model,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        softmax_temperature=args.temperature,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"\nProbe Configuration:")
    print(f"  Target model: {probe_config.target_model}")
    print(f"  Layers: {probe_config.num_layers}")
    print(f"  Input dim: {probe_config.probe_input_dim}")
    print(f"  Device: {args.device}")

    # Load datasets
    print(f"\nLoading data...")
    train_dataset = ActivationDataset.from_json(
        args.train_data,
        activations_dir=args.activations_dir,
    )
    val_dataset = ActivationDataset.from_json(
        args.val_data,
        activations_dir=args.activations_dir,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
    )

    # Create probe and trainer
    probe = LinearActivationProbe(probe_config)
    print(f"\nProbe parameters: {probe.get_num_parameters():,}")

    trainer = ProbeTrainer(
        probe=probe,
        config=probe_config,
        device=args.device,
        use_amp=not args.no_amp,
        loss_type=args.loss_type,
    )

    # Train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "probe.pt"

    print(f"\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=str(save_path),
    )

    # Final evaluation
    print(f"\nFinal Evaluation:")
    final_metrics = trainer.evaluate(val_loader)
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved probe to: {save_path}")
    print(f"Saved history to: {history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
