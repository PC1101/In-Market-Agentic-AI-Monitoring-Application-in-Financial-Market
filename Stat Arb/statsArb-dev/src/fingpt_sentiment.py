"""
FinGPT sentiment inference engine for regime stress detection.

Loads a 4-bit quantized LLaMA2 base model with a FinGPT LoRA adapter
and runs batched sentiment inference on financial headlines.

GPU requirements:
  - 7B model (default): ~6 GB VRAM with 4-bit quantization (bitsandbytes)
  - 13B model: ~12 GB VRAM

LOOK-AHEAD BIAS WARNING:
  FinGPT is trained on financial text that includes post-crisis retrospectives
  (Wikipedia GFC articles, post-mortems, etc.). When scoring historical
  crisis-period headlines, the model may produce more extreme negative
  sentiment than market participants felt at the time — because it "knows"
  what happened next. This is training-data contamination and cannot be
  fully eliminated via prompting. The compute_stress.py diagnostic chart
  (FinGPT stress vs contemporaneous VIX) helps quantify this bias.

  The prompt template below attempts to constrain inference to literal
  headline content, but treat pre-2020 crisis scores with caution.
"""
import os
from typing import Dict, List

import numpy as np

FINGPT_BASE_DEFAULT = "meta-llama/Llama-2-7b-hf"
FINGPT_LORA_DEFAULT = "FinGPT/fingpt-sentiment_llama2-7b_lora"

# Map FinGPT output labels to numeric sentiment scores in [-1, +1]
SENTIMENT_SCORES: Dict[str, float] = {
    "strong negative":     -1.00,
    "moderately negative": -0.67,
    "mildly negative":     -0.33,
    "neutral":              0.00,
    "mildly positive":     +0.33,
    "moderately positive": +0.67,
    "strong positive":     +1.00,
}

# FinGPT standard prompt with an added look-ahead constraint.
# The model is instructed to evaluate only the literal headline content.
_PROMPT_TEMPLATE = (
    "Instruction: What is the sentiment of this news? Please choose an answer from "
    "{strong negative/moderately negative/mildly negative/neutral/mildly positive/"
    "moderately positive/strong positive}. "
    "Evaluate based only on what is stated in the headline itself; "
    "do not use knowledge of events that occurred after its publication date.\n"
    "Input: {headline}\n"
    "Answer: "
)

_DEFAULT_BATCH_SIZE = int(os.environ.get("FINGPT_BATCH_SIZE", "8"))


class FinGPTSentimentEngine:
    """
    Sentiment scoring engine backed by FinGPT (LLaMA2 + LoRA adapter).

    Parameters
    ----------
    base_model : str
        HuggingFace model ID for the LLaMA2 base model.
    lora_model : str
        HuggingFace model ID for the FinGPT LoRA adapter.
    device : str
        "cuda" (recommended) or "cpu" (very slow, for testing only).
    """

    def __init__(
        self,
        base_model: str = FINGPT_BASE_DEFAULT,
        lora_model: str = FINGPT_LORA_DEFAULT,
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.lora_model = lora_model
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                f"Missing dependency: {exc}. "
                "Install with: pip install torch transformers peft accelerate bitsandbytes"
            ) from exc

        print(f"[FinGPT] Loading base model : {self.base_model}")
        print(f"[FinGPT] LoRA adapter       : {self.lora_model}")
        print(f"[FinGPT] Device             : {self.device}")
        print("[FinGPT] (First run downloads ~14 GB from HuggingFace)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base, self.lora_model)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        print("[FinGPT] Model loaded successfully.\n")

    def score_batch(self, headlines: List[str]) -> List[Dict[str, float]]:
        """
        Score a list of headlines, returning one sentiment dict per headline.

        Returns
        -------
        list of dict
            Each dict has all SENTIMENT_SCORES keys. The matched label has
            value 1.0; others 0.0. Falls back to {"neutral": 1.0} on
            parse failure.
        """
        import torch

        prompts = [_PROMPT_TEMPLATE.format(headline=h) for h in headlines]
        results: List[Dict[str, float]] = []

        for i in range(0, len(prompts), _DEFAULT_BATCH_SIZE):
            batch = prompts[i : i + _DEFAULT_BATCH_SIZE]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                new_tokens = output[input_len:]
                decoded = self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip().lower()
                label = self._parse_label(decoded)
                results.append(
                    {k: (1.0 if k == label else 0.0) for k in SENTIMENT_SCORES}
                )

        return results

    def daily_stress(self, headlines: List[str]) -> float:
        """
        Aggregate headlines into a sector-level daily stress score.

        Stress = fraction of headlines whose net numeric sentiment is < 0
        (i.e. the net-negative fraction of the day's headlines).

        Returns
        -------
        float in [0, 1], or np.nan if headlines list is empty.
        """
        if not headlines:
            return np.nan

        batch_results = self.score_batch(headlines)
        numeric_scores = [
            sum(SENTIMENT_SCORES[lbl] * prob for lbl, prob in result.items())
            for result in batch_results
        ]
        n_negative = sum(1 for s in numeric_scores if s < 0)
        return n_negative / len(numeric_scores)

    @staticmethod
    def _parse_label(decoded: str) -> str:
        """Match decoded model output to the closest known sentiment label."""
        decoded = decoded.strip().lower()
        # Exact match first (longest labels first to avoid partial collisions)
        for label in sorted(SENTIMENT_SCORES, key=len, reverse=True):
            if label in decoded:
                return label
        # Partial fallback
        if "negative" in decoded:
            return "moderately negative"
        if "positive" in decoded:
            return "moderately positive"
        return "neutral"
