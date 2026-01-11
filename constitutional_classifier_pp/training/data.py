"""Dataset utilities for Constitutional Classifier++ training.

This module provides:
- ActivationDataset: PyTorch dataset for pre-extracted activations
- On-the-fly activation extraction utilities
- Data loading and preprocessing functions
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Iterator
import json
from pathlib import Path
from tqdm import tqdm


class ActivationDataset(Dataset):
    """Dataset for pre-extracted activations with labels.

    Expects data in format:
    {
        "activations": tensor or path to tensor file,
        "label": int (0 or 1),
        "user_input": str (optional),
        "model_output": str (optional),
    }

    Example:
        >>> dataset = ActivationDataset.from_json("train.json", activations_dir="./acts")
        >>> loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
    """

    def __init__(
        self,
        data: List[Dict],
        max_seq_length: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data: List of samples with activations and labels
            max_seq_length: Max sequence length (truncate if longer)
        """
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Load activations
        activations = sample["activations"]
        if isinstance(activations, str):
            # Load from file
            activations = torch.load(activations)
        elif isinstance(activations, list):
            activations = torch.tensor(activations)

        # Handle different shapes
        if activations.dim() == 2:
            # [seq_len, hidden_dim] -> add batch
            activations = activations.unsqueeze(0)
        elif activations.dim() == 1:
            # [hidden_dim] -> add batch and seq
            activations = activations.unsqueeze(0).unsqueeze(0)

        # Truncate if needed
        if self.max_seq_length and activations.shape[1] > self.max_seq_length:
            activations = activations[:, :self.max_seq_length, :]

        # Create mask
        seq_len = activations.shape[1]
        mask = torch.ones(seq_len, dtype=torch.bool)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "activations": activations.squeeze(0),  # [seq_len, hidden_dim]
            "labels": label,
            "mask": mask,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader with padding.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched tensors with padding
        """
        # Find max sequence length in batch
        max_len = max(sample["activations"].shape[0] for sample in batch)
        hidden_dim = batch[0]["activations"].shape[-1]

        batch_size = len(batch)

        # Initialize padded tensors
        activations = torch.zeros(batch_size, max_len, hidden_dim)
        labels = torch.zeros(batch_size, dtype=torch.long)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, sample in enumerate(batch):
            seq_len = sample["activations"].shape[0]
            activations[i, :seq_len] = sample["activations"]
            labels[i] = sample["labels"]
            masks[i, :seq_len] = True

        return {
            "activations": activations,
            "labels": labels,
            "mask": masks,
        }

    @classmethod
    def from_json(
        cls,
        json_path: str,
        activations_dir: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> "ActivationDataset":
        """Load dataset from JSON file.

        Args:
            json_path: Path to JSON file with sample metadata
            activations_dir: Directory containing activation files
            max_seq_length: Max sequence length

        Returns:
            ActivationDataset instance
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Resolve activation paths
        if activations_dir:
            activations_dir = Path(activations_dir)
            for sample in data:
                if isinstance(sample.get("activations"), str):
                    sample["activations"] = str(activations_dir / sample["activations"])

        return cls(data, max_seq_length=max_seq_length)

    @classmethod
    def from_tensors(
        cls,
        activations: torch.Tensor,
        labels: torch.Tensor,
    ) -> "ActivationDataset":
        """Create dataset from tensors.

        Args:
            activations: [num_samples, seq_len, hidden_dim]
            labels: [num_samples]

        Returns:
            ActivationDataset instance
        """
        data = [
            {"activations": activations[i], "label": labels[i].item()}
            for i in range(len(labels))
        ]
        return cls(data)


def extract_activations_from_model(
    model,
    tokenizer,
    texts: List[str],
    collector,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    show_progress: bool = True,
) -> List[torch.Tensor]:
    """Extract activations from model for a list of texts.

    This extracts activations on-the-fly without saving to disk.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of input texts
        collector: MultiLayerActivationCollector instance
        batch_size: Batch size for extraction
        max_length: Max sequence length
        device: Device to use
        show_progress: Show progress bar

    Returns:
        List of activation tensors [seq_len, hidden_dim * num_layers]
    """
    model.eval()
    all_activations = []

    # Register hooks
    collector.register_hooks()

    try:
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting activations")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Forward pass
            with torch.no_grad():
                model(**inputs)

            # Get activations
            acts = collector.get_concatenated_activations()

            # Store per-sample (remove padding based on attention mask)
            for j, mask in enumerate(inputs.attention_mask):
                seq_len = mask.sum().item()
                sample_acts = acts[j, :seq_len, :].cpu()
                all_activations.append(sample_acts)

            collector.clear()

    finally:
        collector.remove_hooks()

    return all_activations


def create_exchange_dataset(
    model,
    tokenizer,
    exchanges: List[Dict[str, str]],
    labels: List[int],
    collector,
    chat_template: bool = True,
    **kwargs,
) -> ActivationDataset:
    """Create dataset from user/assistant exchanges.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        exchanges: List of {"user": str, "assistant": str} dicts
        labels: Binary labels for each exchange
        collector: MultiLayerActivationCollector
        chat_template: Use chat template for formatting
        **kwargs: Passed to extract_activations_from_model

    Returns:
        ActivationDataset ready for training
    """
    # Format exchanges
    texts = []
    for ex in exchanges:
        if chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = f"User: {ex['user']}\nAssistant: {ex['assistant']}"
        texts.append(text)

    # Extract activations
    activations = extract_activations_from_model(
        model, tokenizer, texts, collector, **kwargs
    )

    # Create dataset
    data = [
        {
            "activations": act,
            "label": label,
            "user_input": ex["user"],
            "model_output": ex["assistant"],
        }
        for act, label, ex in zip(activations, labels, exchanges)
    ]

    return ActivationDataset(data)


class StreamingActivationDataset:
    """Memory-efficient dataset that loads activations on demand.

    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        metadata_path: str,
        activations_dir: str,
        max_seq_length: Optional[int] = None,
    ):
        """Initialize streaming dataset.

        Args:
            metadata_path: Path to JSON with sample metadata
            activations_dir: Directory containing activation .pt files
            max_seq_length: Max sequence length
        """
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.activations_dir = Path(activations_dir)
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]

        # Load activation from disk
        act_path = self.activations_dir / meta["activation_file"]
        activations = torch.load(act_path)

        # Truncate if needed
        if self.max_seq_length and activations.shape[0] > self.max_seq_length:
            activations = activations[: self.max_seq_length]

        seq_len = activations.shape[0]
        mask = torch.ones(seq_len, dtype=torch.bool)

        return {
            "activations": activations,
            "labels": torch.tensor(meta["label"], dtype=torch.long),
            "mask": mask,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Same collate function as ActivationDataset."""
        return ActivationDataset.collate_fn(batch)


def prepare_wildchat_data(
    raw_data_path: str,
    output_path: str,
    model,
    tokenizer,
    collector,
    sample_size: Optional[int] = None,
    harmful_keywords: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Prepare WildChat data for probe training.

    Processes WildChat conversations, extracts activations, and saves
    in format suitable for ActivationDataset.

    Args:
        raw_data_path: Path to raw WildChat JSON
        output_path: Path to save processed dataset
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        collector: MultiLayerActivationCollector
        sample_size: Limit number of samples
        harmful_keywords: Keywords to identify potentially harmful content
        **kwargs: Passed to extract_activations_from_model
    """
    import random

    # Load raw data
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    if sample_size and len(raw_data) > sample_size:
        raw_data = random.sample(raw_data, sample_size)

    # Default harmful keywords
    if harmful_keywords is None:
        harmful_keywords = ["hack", "exploit", "attack", "malware", "illegal"]

    # Process and label
    exchanges = []
    labels = []

    for item in raw_data:
        # Extract user and assistant turns
        user = item.get("user", item.get("input", ""))
        assistant = item.get("assistant", item.get("output", ""))

        if not user or not assistant:
            continue

        exchanges.append({"user": user, "assistant": assistant})

        # Simple heuristic labeling (replace with proper labels in production)
        is_harmful = any(
            kw.lower() in user.lower() or kw.lower() in assistant.lower()
            for kw in harmful_keywords
        )
        labels.append(1 if is_harmful else 0)

    print(f"Processed {len(exchanges)} exchanges ({sum(labels)} harmful)")

    # Create dataset
    dataset = create_exchange_dataset(
        model, tokenizer, exchanges, labels, collector, **kwargs
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = []
    activations_dir = output_path.parent / "activations"
    activations_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(dataset.data):
        act_file = f"sample_{i:06d}.pt"
        torch.save(sample["activations"], activations_dir / act_file)

        metadata.append({
            "activation_file": act_file,
            "label": sample["label"],
            "user_input": sample.get("user_input", ""),
            "model_output": sample.get("model_output", ""),
        })

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {output_path}")
