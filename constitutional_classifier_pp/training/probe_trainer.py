"""Training utilities for Stage 1 Linear Activation Probe.

This module implements:
- SoftmaxWeightedBCELoss: Focuses training on "confidently harmful" tokens
- SlidingWindowMeanLoss: Alternative using sliding window averaging
- ProbeTrainer: Full training loop with evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Callable
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config.base import ProbeConfig
from models.probe import LinearActivationProbe


class SoftmaxWeightedBCELoss(nn.Module):
    """Softmax-weighted BCE loss for probe training.

    Loss = sum_t [w_t * BCE(y, sigmoid(z_bar_t))]
    where w_t = softmax(z_bar_t / tau)

    This assigns higher weight to tokens where the probe is "confident",
    focusing training on the most informative tokens and ignoring harmless
    prefixes. The softmax weighting penalizes confident false positives
    while allowing the probe to ignore tokens with negative logits.

    Args:
        temperature: Softmax temperature (tau). Lower = sharper focus on max
        reduction: How to reduce batch dimension ("mean" or "sum")
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute softmax-weighted BCE loss.

        Args:
            logits: [batch, seq_len, 1] or [batch, seq_len] - Raw logits
            labels: [batch] - Exchange-level binary labels (0 or 1)
            mask: [batch, seq_len] - Optional attention mask (1 = valid, 0 = pad)

        Returns:
            Weighted loss scalar
        """
        # Handle shape
        if logits.dim() == 3:
            logits = logits.squeeze(-1)  # [batch, seq_len]

        batch_size, seq_len = logits.shape

        # Apply mask if provided
        if mask is not None:
            # Mask out padding with large negative value
            logits = logits.masked_fill(~mask.bool(), float("-inf"))

        # Compute softmax weights: w_t = exp(z_t/tau) / sum(exp(z_t'/tau))
        weights = F.softmax(logits / self.temperature, dim=1)  # [batch, seq_len]

        # Expand labels to sequence length
        labels_expanded = labels.unsqueeze(1).expand(-1, seq_len).float()  # [batch, seq_len]

        # BCE at each position with exchange-level label
        bce_per_position = F.binary_cross_entropy_with_logits(
            logits, labels_expanded, reduction="none"
        )  # [batch, seq_len]

        # Handle -inf in logits (masked positions)
        bce_per_position = torch.where(
            torch.isinf(logits),
            torch.zeros_like(bce_per_position),
            bce_per_position
        )

        # Weighted sum across sequence
        weighted_loss = (weights * bce_per_position).sum(dim=1)  # [batch]

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class SlidingWindowMeanLoss(nn.Module):
    """BCE loss with sliding window mean for logits.

    Uses z_bar_t = (1/M) * sum_{k=0}^{M-1} z_{t-k} for smoothed logits,
    then applies softmax weighting on the smoothed values.

    Args:
        window_size: Size of sliding window M (default 16)
        temperature: Softmax temperature for weighting
        reduction: How to reduce batch dimension
    """

    def __init__(
        self,
        window_size: int = 16,
        temperature: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute sliding window BCE loss.

        Args:
            logits: [batch, seq_len, 1] or [batch, seq_len]
            labels: [batch]
            mask: [batch, seq_len] (optional)

        Returns:
            Loss scalar
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        batch_size, seq_len = logits.shape

        # Apply mask before windowing
        if mask is not None:
            logits = logits * mask.float()

        # Pad for valid sliding window (left padding)
        padded = F.pad(logits, (self.window_size - 1, 0), value=0.0)

        # Compute sliding window mean using unfold
        windows = padded.unfold(1, self.window_size, 1)  # [batch, seq_len, window]
        z_bar = windows.mean(dim=2)  # [batch, seq_len]

        # Softmax weights on z_bar
        if mask is not None:
            z_bar = z_bar.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(z_bar / self.temperature, dim=1)

        # BCE loss on z_bar
        labels_expanded = labels.unsqueeze(1).expand(-1, seq_len).float()
        bce_per_position = F.binary_cross_entropy_with_logits(
            z_bar, labels_expanded, reduction="none"
        )

        # Handle masked positions
        bce_per_position = torch.where(
            torch.isinf(z_bar),
            torch.zeros_like(bce_per_position),
            bce_per_position
        )

        weighted_loss = (weights * bce_per_position).sum(dim=1)

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class ProbeTrainer:
    """Trainer for Stage 1 Linear Activation Probe.

    Handles the full training loop including:
    - Gradient accumulation
    - Mixed precision training
    - Evaluation with AUROC, F1, etc.
    - Threshold calibration
    - Checkpointing

    Example:
        >>> trainer = ProbeTrainer(probe, config)
        >>> trainer.train(train_loader, val_loader, num_epochs=10)
        >>> metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        probe: LinearActivationProbe,
        config: ProbeConfig,
        device: str = "cuda",
        use_amp: bool = True,
        loss_type: str = "softmax_weighted",  # or "sliding_window"
    ):
        self.probe = probe.to(device)
        self.config = config
        self.device = device
        self.use_amp = use_amp and device == "cuda"

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        if loss_type == "softmax_weighted":
            self.loss_fn = SoftmaxWeightedBCELoss(
                temperature=config.softmax_temperature
            )
        elif loss_type == "sliding_window":
            self.loss_fn = SlidingWindowMeanLoss(
                window_size=config.window_size,
                temperature=config.softmax_temperature,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Training state
        self.global_step = 0
        self.best_auroc = 0.0

    def train_epoch(
        self,
        dataloader: DataLoader,
        accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            accumulation_steps: Gradient accumulation steps

        Returns:
            Dict with training metrics (loss, etc.)
        """
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(pbar):
            activations = batch["activations"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass with optional AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.probe(activations)
                loss = self.loss_fn(logits, labels, mask)
                loss = loss / accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        return {"train_loss": total_loss / num_batches}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate probe on validation/test data.

        Computes:
        - AUROC
        - Best F1 score and corresponding threshold
        - Precision/Recall at best F1

        Args:
            dataloader: Validation/test data loader

        Returns:
            Dict with evaluation metrics
        """
        self.probe.eval()
        all_scores: List[float] = []
        all_labels: List[int] = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                activations = batch["activations"].to(self.device)
                labels = batch["labels"].to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)

                logits = self.probe(activations)
                loss = self.loss_fn(logits, labels, mask)
                total_loss += loss.item()
                num_batches += 1

                # Get max probability per sequence for ranking
                probs = torch.sigmoid(logits)
                if probs.dim() == 3:
                    probs = probs.squeeze(-1)
                max_probs = probs.max(dim=1)[0]  # [batch]

                all_scores.extend(max_probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        metrics = self._compute_metrics(all_scores, all_labels)
        metrics["eval_loss"] = total_loss / num_batches

        return metrics

    def _compute_metrics(
        self,
        scores: List[float],
        labels: List[int]
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        try:
            from sklearn.metrics import (
                roc_auc_score,
                precision_recall_curve,
                average_precision_score,
            )

            scores_np = np.array(scores)
            labels_np = np.array(labels)

            # AUROC
            auroc = roc_auc_score(labels_np, scores_np)

            # Average precision
            ap = average_precision_score(labels_np, scores_np)

            # Best F1 threshold
            precision, recall, thresholds = precision_recall_curve(labels_np, scores_np)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_f1_idx = np.argmax(f1_scores)

            best_f1 = f1_scores[best_f1_idx]
            best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
            precision_at_best = precision[best_f1_idx]
            recall_at_best = recall[best_f1_idx]

            return {
                "auroc": float(auroc),
                "average_precision": float(ap),
                "best_f1": float(best_f1),
                "best_threshold": float(best_threshold),
                "precision_at_best_f1": float(precision_at_best),
                "recall_at_best_f1": float(recall_at_best),
            }

        except ImportError:
            # Fallback without sklearn
            return {"auroc": 0.0, "note": "sklearn not available"}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 3,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_path: Path to save best checkpoint
            early_stopping_patience: Epochs without improvement before stopping

        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "auroc": [],
        }
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(
                train_loader,
                accumulation_steps=self.config.gradient_accumulation_steps,
            )

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Log
            print(
                f"  Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['eval_loss']:.4f} | "
                f"AUROC: {val_metrics['auroc']:.4f} | "
                f"Best F1: {val_metrics['best_f1']:.4f} @ {val_metrics['best_threshold']:.3f}"
            )

            # Update history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["eval_loss"])
            history["auroc"].append(val_metrics["auroc"])

            # Save best model
            if val_metrics["auroc"] > self.best_auroc:
                self.best_auroc = val_metrics["auroc"]
                patience_counter = 0
                if save_path:
                    self.probe.save_pretrained(save_path)
                    print(f"  Saved best model (AUROC: {self.best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping after {patience_counter} epochs without improvement")
                    break

        return history

    def calibrate_threshold(
        self,
        dataloader: DataLoader,
        target_rate: float = 0.055,
    ) -> float:
        """Calibrate T1 threshold for target escalation rate.

        Args:
            dataloader: Calibration data (production-like benign traffic)
            target_rate: Target fraction of traffic to escalate

        Returns:
            Calibrated threshold
        """
        self.probe.eval()
        all_scores: List[float] = []

        with torch.no_grad():
            for batch in dataloader:
                activations = batch["activations"].to(self.device)
                logits = self.probe(activations)
                probs = torch.sigmoid(logits)
                if probs.dim() == 3:
                    probs = probs.squeeze(-1)
                max_probs = probs.max(dim=1)[0]
                all_scores.extend(max_probs.cpu().tolist())

        # Sort descending and find threshold
        all_scores.sort(reverse=True)
        threshold_idx = int(len(all_scores) * target_rate)
        threshold = all_scores[min(threshold_idx, len(all_scores) - 1)]

        print(f"Calibrated T1 threshold: {threshold:.4f} for {target_rate * 100:.1f}% escalation rate")

        return threshold
