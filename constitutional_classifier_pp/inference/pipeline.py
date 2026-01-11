"""Production Inference Pipeline for Constitutional Classifier++.

This module provides the main interface for integrating the two-stage
classifier into a production LLM serving system, with support for
streaming generation and real-time safety monitoring.
"""

import torch
from typing import Dict, Optional, Generator, List, Callable, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config.base import ProbeConfig, ClassifierConfig, EnsembleConfig
from models.probe import LinearActivationProbe
from models.classifier import ExternalClassifier
from models.ensemble import TwoStageEnsemble, EnsembleResult
from hooks.activation_collector import MultiLayerActivationCollector, StreamingActivationCollector


@dataclass
class GenerationResult:
    """Result from generate_with_safety."""

    output: str  # Generated text
    safety_result: EnsembleResult  # Final safety classification
    token_scores: List[float]  # Per-token probe scores
    max_probe_score: float  # Maximum probe score during generation
    stopped_early: bool = False  # Whether generation was stopped due to safety
    tokens_generated: int = 0  # Number of tokens generated


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""

    probe_config: ProbeConfig = field(default_factory=ProbeConfig)
    classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False

    # Safety options
    early_stop_on_unsafe: bool = True
    early_stop_threshold: Optional[float] = None  # None = use t2_threshold
    check_every_n_tokens: int = 1  # Check safety every N tokens

    # Streaming
    stream_tokens: bool = False


class ConstitutionalClassifierPipeline:
    """Production inference pipeline for Constitutional Classifier++.

    Integrates the two-stage classifier into LLM generation with:
    - Real-time safety monitoring via activation probing
    - Optional early stopping when unsafe content detected
    - Streaming token-by-token generation support
    - Post-hoc classification of pre-generated exchanges

    Example:
        >>> pipeline = ConstitutionalClassifierPipeline.from_pretrained(
        ...     model_path="meta-llama/Llama-3.1-8B-Instruct",
        ...     probe_path="./probe.pt",
        ...     adapter_path="./classifier_lora"
        ... )
        >>> result = pipeline.generate_with_safety("What is 2+2?")
        >>> print(result.output)
        >>> print(f"Safe: {not result.safety_result.should_refuse}")
    """

    def __init__(
        self,
        target_model,
        tokenizer,
        probe: LinearActivationProbe,
        classifier: ExternalClassifier,
        config: PipelineConfig,
        device: Optional[str] = None,
    ):
        """Initialize the pipeline.

        Args:
            target_model: HuggingFace model to protect
            tokenizer: HuggingFace tokenizer
            probe: Trained Stage 1 probe
            classifier: Stage 2 classifier
            config: Pipeline configuration
            device: Device to use
        """
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.probe = probe
        self.classifier = classifier
        self.config = config
        self.device = device or next(target_model.parameters()).device

        # Set up ensemble
        self.ensemble = TwoStageEnsemble(
            probe, classifier, config.ensemble_config
        )

        # Set up activation collector
        self._setup_collector()

        # Early stop threshold
        self.early_stop_threshold = (
            config.early_stop_threshold or config.ensemble_config.t2_threshold
        )

    def _setup_collector(self):
        """Set up activation collectors for the target model."""
        probe_config = self.config.probe_config

        # Standard collector for post-hoc classification
        self.collector = MultiLayerActivationCollector(
            model=self.target_model,
            layers=probe_config.layers_to_probe,
            hook_point=probe_config.hook_point,
        )

        # Streaming collector for real-time monitoring
        self.streaming_collector = StreamingActivationCollector(
            model=self.target_model,
            layers=probe_config.layers_to_probe,
            hook_point=probe_config.hook_point,
        )

    def generate_with_safety(
        self,
        user_input: str,
        max_new_tokens: Optional[int] = None,
        early_stop_on_unsafe: Optional[bool] = None,
        **generate_kwargs,
    ) -> GenerationResult:
        """Generate response with real-time safety monitoring.

        Monitors model activations token-by-token during generation,
        optionally stopping early if unsafe content is detected.

        Args:
            user_input: User's prompt
            max_new_tokens: Maximum tokens to generate
            early_stop_on_unsafe: Stop generation if unsafe detected
            **generate_kwargs: Additional arguments for model.generate()

        Returns:
            GenerationResult with output, safety classification, and scores
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        early_stop = early_stop_on_unsafe if early_stop_on_unsafe is not None else self.config.early_stop_on_unsafe

        # Prepare input
        messages = [{"role": "user", "content": user_input}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"User: {user_input}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # Initialize tracking
        token_scores: List[float] = []
        max_score = 0.0
        stopped_early = False

        # Reset probe EMA state
        self.probe.reset_ema()

        # Register streaming hooks
        self.streaming_collector.register_hooks()
        self.streaming_collector.reset()

        try:
            # Token-by-token generation
            generated_ids = inputs.input_ids.clone()
            attention_mask = inputs.attention_mask.clone()

            for step in range(max_new_tokens):
                # Forward pass
                with torch.no_grad():
                    outputs = self.target_model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]

                if generate_kwargs.get("do_sample", self.config.do_sample):
                    # Sampling
                    temp = generate_kwargs.get("temperature", self.config.temperature)
                    top_p = generate_kwargs.get("top_p", self.config.top_p)
                    probs = torch.softmax(next_token_logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Get activations and probe score
                if (step + 1) % self.config.check_every_n_tokens == 0:
                    acts = self.streaming_collector.get_concatenated_activations()
                    probe_output = self.probe.forward_streaming(acts, reset=(step == 0))
                    score = probe_output.max_probability
                    token_scores.append(score)
                    max_score = max(max_score, score)

                    # Early stopping check
                    if early_stop and score > self.early_stop_threshold:
                        stopped_early = True
                        break

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update for next iteration
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                self.streaming_collector.clear()

        finally:
            self.streaming_collector.remove_hooks()

        # Decode output
        output_ids = generated_ids[0, prompt_length:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Final classification with full context
        safety_result = self._classify_exchange(user_input, output_text, generated_ids)

        return GenerationResult(
            output=output_text,
            safety_result=safety_result,
            token_scores=token_scores,
            max_probe_score=max_score,
            stopped_early=stopped_early,
            tokens_generated=len(output_ids),
        )

    def _classify_exchange(
        self,
        user_input: str,
        model_output: str,
        full_input_ids: Optional[torch.Tensor] = None,
    ) -> EnsembleResult:
        """Classify an exchange using full context.

        Args:
            user_input: User's message
            model_output: Model's response
            full_input_ids: Optional pre-tokenized full exchange

        Returns:
            EnsembleResult from two-stage classification
        """
        if full_input_ids is None:
            # Tokenize the exchange
            exchange = f"User: {user_input}\nAssistant: {model_output}"
            inputs = self.tokenizer(
                exchange,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)
        else:
            inputs = {"input_ids": full_input_ids}

        # Get activations
        self.collector.register_hooks()
        try:
            with torch.no_grad():
                self.target_model(**inputs)
            activations = self.collector.get_concatenated_activations()
        finally:
            self.collector.remove_hooks()

        return self.ensemble.classify(activations, user_input, model_output)

    def classify_exchange(
        self,
        user_input: str,
        model_output: str,
    ) -> EnsembleResult:
        """Classify a pre-generated exchange.

        Use this for post-hoc classification of existing conversations.

        Args:
            user_input: User's message
            model_output: AI's response

        Returns:
            EnsembleResult
        """
        return self._classify_exchange(user_input, model_output)

    def batch_classify(
        self,
        exchanges: List[Dict[str, str]],
        batch_size: int = 8,
    ) -> List[EnsembleResult]:
        """Classify multiple exchanges in batches.

        Args:
            exchanges: List of {"user": str, "assistant": str} dicts
            batch_size: Batch size for processing

        Returns:
            List of EnsembleResult
        """
        results = []

        for i in range(0, len(exchanges), batch_size):
            batch = exchanges[i:i + batch_size]

            for ex in batch:
                result = self.classify_exchange(ex["user"], ex["assistant"])
                results.append(result)

        return results

    def stream_generate_with_safety(
        self,
        user_input: str,
        max_new_tokens: Optional[int] = None,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream tokens with safety scores.

        Yields each token as it's generated along with the current safety score.

        Args:
            user_input: User's prompt
            max_new_tokens: Maximum tokens
            callback: Optional callback(token_text, safety_score) called per token

        Yields:
            Dict with "token", "score", "should_stop"
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # Prepare input
        messages = [{"role": "user", "content": user_input}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"User: {user_input}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Reset state
        self.probe.reset_ema()
        self.streaming_collector.register_hooks()
        self.streaming_collector.reset()

        generated_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()

        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.target_model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Get safety score
                acts = self.streaming_collector.get_concatenated_activations()
                probe_output = self.probe.forward_streaming(acts, reset=(step == 0))
                score = probe_output.max_probability

                # Decode token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)

                should_stop = (
                    next_token.item() == self.tokenizer.eos_token_id or
                    (self.config.early_stop_on_unsafe and score > self.early_stop_threshold)
                )

                if callback:
                    callback(token_text, score)

                yield {
                    "token": token_text,
                    "score": score,
                    "should_stop": should_stop,
                    "step": step,
                }

                if should_stop:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                self.streaming_collector.clear()

        finally:
            self.streaming_collector.remove_hooks()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        probe_path: str,
        adapter_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        **model_kwargs,
    ) -> "ConstitutionalClassifierPipeline":
        """Load a complete pipeline from pretrained components.

        Args:
            model_path: Path/name of target HuggingFace model
            probe_path: Path to trained probe weights
            adapter_path: Path to Stage 2 classifier LoRA adapter
            config: Optional pipeline config
            device: Device to use
            load_in_4bit: Load target model in 4-bit
            **model_kwargs: Additional args for model loading

        Returns:
            Initialized ConstitutionalClassifierPipeline
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = config or PipelineConfig()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading target model: {model_path}")

        # Load target model
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            target_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                **model_kwargs,
            )
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **model_kwargs,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load probe
        print(f"Loading probe: {probe_path}")
        probe = LinearActivationProbe.from_pretrained(probe_path, config.probe_config)
        probe = probe.to(device)
        probe.eval()

        # Load classifier
        print("Initializing Stage 2 classifier")
        classifier_config = config.classifier_config
        if adapter_path:
            classifier_config.adapter_path = adapter_path
        classifier = ExternalClassifier(classifier_config)

        return cls(
            target_model=target_model,
            tokenizer=tokenizer,
            probe=probe,
            classifier=classifier,
            config=config,
            device=device,
        )

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return self.ensemble.get_statistics()

    def __repr__(self) -> str:
        return (
            f"ConstitutionalClassifierPipeline("
            f"model={type(self.target_model).__name__}, "
            f"ensemble={self.ensemble})"
        )
