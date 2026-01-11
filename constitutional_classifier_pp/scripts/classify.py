#!/usr/bin/env python3
"""Classify exchanges using Constitutional Classifier++.

Example usage:
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
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from constitutional_classifier_pp.config.base import (
    ProbeConfig,
    ClassifierConfig,
    EnsembleConfig,
)
from constitutional_classifier_pp.models.probe import LinearActivationProbe
from constitutional_classifier_pp.models.classifier import ExternalClassifier, MockClassifier
from constitutional_classifier_pp.models.ensemble import TwoStageEnsemble


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify exchanges using Constitutional Classifier++"
    )

    # Models
    parser.add_argument(
        "--probe",
        type=str,
        required=True,
        help="Path to trained probe weights",
    )
    parser.add_argument(
        "--classifier-adapter",
        type=str,
        default=None,
        help="Path to classifier LoRA adapter (optional)",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Classifier base model",
    )
    parser.add_argument(
        "--mock-classifier",
        action="store_true",
        help="Use mock classifier (for testing without GPU)",
    )

    # Input options
    parser.add_argument(
        "--user",
        type=str,
        help="User message to classify",
    )
    parser.add_argument(
        "--assistant",
        type=str,
        help="Assistant response to classify",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="JSON file with exchanges to classify",
    )

    # Thresholds
    parser.add_argument(
        "--t1",
        type=float,
        default=0.3,
        help="Stage 1 escalation threshold",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=0.5,
        help="Stage 2 refusal threshold",
    )

    # Output
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for batch results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    return parser.parse_args()


def create_synthetic_activations(probe_config: ProbeConfig, seq_len: int = 50):
    """Create synthetic activations for testing.

    In production, these would come from the target model's forward pass.
    """
    return torch.randn(1, seq_len, probe_config.probe_input_dim)


def classify_single(
    ensemble: TwoStageEnsemble,
    probe_config: ProbeConfig,
    user: str,
    assistant: str,
    verbose: bool = False,
):
    """Classify a single exchange."""
    # Create synthetic activations (in production, extract from model)
    activations = create_synthetic_activations(probe_config)

    result = ensemble.classify(activations, user, assistant)

    if verbose:
        print(f"\nUser: {user[:100]}...")
        print(f"Assistant: {assistant[:100]}...")
        print(f"\nResults:")
        print(f"  Stage 1 Score: {result.stage1_score:.4f}")
        if result.escalated:
            print(f"  Stage 2 Score: {result.stage2_score:.4f}")
        print(f"  Final Score: {result.final_score:.4f}")
        print(f"  Escalated: {result.escalated}")
        print(f"  Should Refuse: {result.should_refuse}")

    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("Constitutional Classifier++ Exchange Classification")
    print("=" * 60)

    # Load probe
    print(f"\nLoading probe from: {args.probe}")
    probe_config = ProbeConfig()  # Use defaults, will be overridden by weights
    probe = LinearActivationProbe.from_pretrained(args.probe, probe_config)
    probe = probe.to(args.device)
    probe.eval()

    # Load or mock classifier
    if args.mock_classifier:
        print("Using mock classifier")
        classifier = MockClassifier(default_probability=0.3)
    else:
        print(f"Loading classifier: {args.classifier_model}")
        classifier_config = ClassifierConfig(
            model_name=args.classifier_model,
            adapter_path=args.classifier_adapter,
        )
        classifier = ExternalClassifier(classifier_config)

    # Create ensemble
    ensemble_config = EnsembleConfig(
        t1_threshold=args.t1,
        t2_threshold=args.t2,
    )
    ensemble = TwoStageEnsemble(probe, classifier, ensemble_config)

    print(f"\nThresholds: T1={args.t1}, T2={args.t2}")

    # Single exchange mode
    if args.user and args.assistant:
        result = classify_single(
            ensemble, probe_config, args.user, args.assistant, verbose=True
        )
        print(f"\n{'='*60}")
        print(f"VERDICT: {'REFUSE' if result.should_refuse else 'ALLOW'}")
        print(f"{'='*60}")
        return

    # Batch mode
    if args.input_file:
        print(f"\nLoading exchanges from: {args.input_file}")
        with open(args.input_file, "r") as f:
            exchanges = json.load(f)

        results = []
        for i, ex in enumerate(exchanges):
            user = ex.get("user", ex.get("user_input", ""))
            assistant = ex.get("assistant", ex.get("model_output", ""))

            result = classify_single(
                ensemble, probe_config, user, assistant, verbose=args.verbose
            )

            results.append({
                "user": user,
                "assistant": assistant,
                "stage1_score": result.stage1_score,
                "stage2_score": result.stage2_score,
                "final_score": result.final_score,
                "escalated": result.escalated,
                "should_refuse": result.should_refuse,
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(exchanges)} exchanges")

        # Summary
        num_refused = sum(r["should_refuse"] for r in results)
        num_escalated = sum(r["escalated"] for r in results)

        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total exchanges: {len(results)}")
        print(f"  Escalated to Stage 2: {num_escalated} ({100*num_escalated/len(results):.1f}%)")
        print(f"  Refused: {num_refused} ({100*num_refused/len(results):.1f}%)")
        print(f"{'='*60}")

        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to: {args.output_file}")

        return

    print("\nError: Provide either --user and --assistant, or --input-file")


if __name__ == "__main__":
    main()
