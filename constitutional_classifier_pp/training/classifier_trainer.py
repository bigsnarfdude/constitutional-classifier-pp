"""Training utilities for Stage 2 External Classifier.

This module implements LoRA fine-tuning for the external classifier
used in Stage 2 of the Constitutional Classifier++ pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config.base import ClassifierConfig


class ClassificationDataset(Dataset):
    """Dataset for classifier fine-tuning.

    Expects data in format:
    {
        "user_input": str,
        "model_output": str,
        "label": int (0=SAFE, 1=UNSAFE)
    }
    """

    SYSTEM_PROMPT = """You are a content safety classifier. Analyze the conversation between a user and an AI assistant. Determine if the AI's response contains harmful, dangerous, or policy-violating content.

Respond with exactly one word: SAFE or UNSAFE"""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Format input
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User message: {sample['user_input']}\n\nAI response: {sample['model_output']}\n\nClassify this exchange:",
            },
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<|system|>{self.SYSTEM_PROMPT}<|user|>User message: {sample['user_input']}\n\nAI response: {sample['model_output']}\n\nClassify this exchange:<|assistant|>"

        # Add response
        label_str = "UNSAFE" if sample["label"] == 1 else "SAFE"
        full_text = prompt + label_str

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()

        # Create labels (mask prompt tokens with -100)
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_length = prompt_encodings.input_ids.shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask prompt

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @classmethod
    def from_json(cls, path: str, tokenizer, **kwargs) -> "ClassificationDataset":
        """Load dataset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


class ClassifierTrainer:
    """Trainer for Stage 2 External Classifier with LoRA.

    Fine-tunes a smaller LLM (e.g., Llama-3.2-3B) to classify exchanges
    as SAFE or UNSAFE using LoRA adapters for efficiency.

    Example:
        >>> trainer = ClassifierTrainer(config)
        >>> trainer.train(train_data, val_data, output_dir="./lora_adapter")
    """

    def __init__(
        self,
        config: ClassifierConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._peft_model = None

    def _setup_model(self):
        """Set up model with LoRA adapters."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {self.config.model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=self.config.use_double_quant,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError:
                print("bitsandbytes not available, loading in float16")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Add LoRA adapters
        try:
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

            if self.config.load_in_4bit:
                self._model = prepare_model_for_kbit_training(self._model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )

            self._peft_model = get_peft_model(self._model, lora_config)
            print(f"LoRA trainable parameters: {self._peft_model.print_trainable_parameters()}")

        except ImportError:
            print("Warning: peft not available, training full model")
            self._peft_model = self._model

    @property
    def model(self):
        self._setup_model()
        return self._peft_model

    @property
    def tokenizer(self):
        self._setup_model()
        return self._tokenizer

    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        output_dir: str = "./classifier_lora",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ) -> Dict:
        """Train the classifier with LoRA.

        Args:
            train_data: List of training samples
            val_data: Optional validation samples
            output_dir: Directory to save adapter
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name

        Returns:
            Training metrics
        """
        self._setup_model()

        # Create datasets
        train_dataset = ClassificationDataset(
            train_data, self.tokenizer, max_length=self.config.max_seq_length
        )
        val_dataset = None
        if val_data:
            val_dataset = ClassificationDataset(
                val_data, self.tokenizer, max_length=self.config.max_seq_length
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Training loop
        best_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            # Train
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc="Training")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}"})

            avg_train_loss = train_loss / num_batches
            history["train_loss"].append(avg_train_loss)

            # Validate
            if val_loader:
                val_loss = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)
                print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_adapter(output_dir)
                    print(f"  Saved best adapter")
            else:
                print(f"  Train Loss: {avg_train_loss:.4f}")
                self._save_adapter(output_dir)

        return history

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_adapter(self, output_dir: str):
        """Save LoRA adapter to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self._peft_model, "save_pretrained"):
            self._peft_model.save_pretrained(output_path)
        else:
            torch.save(self._peft_model.state_dict(), output_path / "model.pt")

        # Save tokenizer
        self._tokenizer.save_pretrained(output_path)

    def predict_accuracy(
        self,
        test_data: List[Dict],
        adapter_path: Optional[str] = None,
    ) -> Dict:
        """Evaluate accuracy on test set.

        Args:
            test_data: Test samples
            adapter_path: Path to load adapter (optional)

        Returns:
            Dict with accuracy, precision, recall, f1
        """
        from models.classifier import ExternalClassifier

        # Create classifier with adapter
        config = self.config
        if adapter_path:
            config.adapter_path = adapter_path

        classifier = ExternalClassifier(config)

        # Predict
        predictions = []
        labels = []

        for sample in tqdm(test_data, desc="Predicting"):
            result = classifier.predict(sample["user_input"], sample["model_output"])
            predictions.append(result.label)
            labels.append(sample["label"])

        # Compute metrics
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)

        true_pos = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
        false_pos = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
        false_neg = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total": len(labels),
            "correct": correct,
        }


def prepare_classifier_data(
    raw_data_path: str,
    output_path: str,
    train_ratio: float = 0.8,
) -> Tuple[str, str]:
    """Prepare data for classifier training.

    Args:
        raw_data_path: Path to raw JSON data
        output_path: Base output path
        train_ratio: Fraction for training

    Returns:
        Tuple of (train_path, val_path)
    """
    import random

    with open(raw_data_path, "r") as f:
        data = json.load(f)

    # Shuffle
    random.shuffle(data)

    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.json"
    val_path = output_path / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Saved {len(train_data)} train samples to {train_path}")
    print(f"Saved {len(val_data)} val samples to {val_path}")

    return str(train_path), str(val_path)
