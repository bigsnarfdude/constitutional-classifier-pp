#!/usr/bin/env python3
"""Train the Stage 2 External Classifier with LoRA.

Example usage:
    python -m constitutional_classifier_pp.scripts.train_classifier \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --train-data data/classifier_train.json \
        --val-data data/classifier_val.json \
        --output-dir outputs/classifier_lora \
        --epochs 3
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from constitutional_classifier_pp.config.base import ClassifierConfig
from constitutional_classifier_pp.training.classifier_trainer import (
    ClassifierTrainer,
    prepare_classifier_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Constitutional Classifier++ Stage 2 Classifier"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model for classifier",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )

    # LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
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
        default=None,
        help="Path to validation data JSON",
    )
    parser.add_argument(
        "--prepare-data",
        type=str,
        default=None,
        help="Path to raw data to prepare (splits into train/val)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/classifier_lora",
        help="Output directory for LoRA adapter",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Constitutional Classifier++ Classifier Training")
    print("=" * 60)

    # Prepare data if requested
    if args.prepare_data:
        print(f"\nPreparing data from: {args.prepare_data}")
        output_dir = Path(args.output_dir).parent / "data"
        train_path, val_path = prepare_classifier_data(
            args.prepare_data,
            str(output_dir / "processed.json"),
        )
        args.train_data = train_path
        args.val_data = val_path

    # Load data
    print(f"\nLoading training data: {args.train_data}")
    with open(args.train_data, "r") as f:
        train_data = json.load(f)
    print(f"  Train samples: {len(train_data)}")

    val_data = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        with open(args.val_data, "r") as f:
            val_data = json.load(f)
        print(f"  Val samples: {len(val_data)}")

    # Create config
    config = ClassifierConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    print(f"\nClassifier Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  4-bit: {config.load_in_4bit}")
    print(f"  Device: {args.device}")

    # Create trainer
    trainer = ClassifierTrainer(config, device=args.device)

    # Train
    print(f"\nStarting training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
    )

    # Save history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved adapter to: {args.output_dir}")
    print(f"Saved history to: {history_path}")

    # Evaluate if we have test data
    if val_data:
        print(f"\nEvaluating on validation set...")
        metrics = trainer.predict_accuracy(val_data, adapter_path=args.output_dir)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
