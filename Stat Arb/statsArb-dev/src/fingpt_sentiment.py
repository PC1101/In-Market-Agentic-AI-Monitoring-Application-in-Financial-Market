"""
Financial headline sentiment engines for regime stress detection.

Two backends are provided:

  FinBERTSentimentEngine  (default, recommended)
    - Model: ProsusAI/finbert  (~400 MB)
    - VRAM: ~500 MB — fits any GPU including 4 GB laptop cards
    - BERT-based encoder, fine-tuned on financial text
    - Direct positive/negative/neutral classification

  FinGPTSentimentEngine   (optional, requires ≥8 GB VRAM)
    - Model: LLaMA2-7B + FinGPT LoRA adapter (~4–5 GB in 4-bit)
    - Generative LLM, richer context understanding
    - Use with --model-type fingpt on a larger GPU

Both expose the same interface: score_batch() and daily_stress().

LOOK-AHEAD BIAS WARNING:
  Both models were trained on financial corpora that may include post-crisis
  text. Inspect results/fingpt_stress_vs_vix_<etf>.png before trusting
  backtest results from crisis periods (GFC 2008, COVID 2020).
"""
import os
from typing import Dict, List

import numpy as np

# ── FinBERT defaults ───────────────────────────────────────────────────────────
FINBERT_MODEL_DEFAULT = "ProsusAI/finbert"

# ── FinGPT defaults (for large-GPU users) ─────────────────────────────────────
FINGPT_BASE_DEFAULT = "meta-llama/Llama-2-7b-hf"
FINGPT_LORA_DEFAULT = "FinGPT/fingpt-sentiment_llama2-7b_lora"

_DEFAULT_BATCH_SIZE = int(os.environ.get("FINGPT_BATCH_SIZE", "16"))


# ══════════════════════════════════════════════════════════════════════════════
# FinBERT  (recommended for ≤8 GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════

class FinBERTSentimentEngine:
    """
    Financial sentiment scoring using ProsusAI/finbert (~400 MB).

    Fits comfortably on a 4 GB laptop GPU. Outputs positive/negative/neutral
    probabilities per headline.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default: ProsusAI/finbert).
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL_DEFAULT,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._pipe = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline, BertTokenizer, BertForSequenceClassification
        except ImportError as exc:
            raise ImportError(
                f"Missing dependency: {exc}. "
                "Install with: pip install transformers torch"
            ) from exc

        print(f"[FinBERT] Loading {self.model_name} on {self.device}...")
        device_idx = 0 if self.device == "cuda" else -1

        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(self.model_name)

        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_idx,
            top_k=None,           # return all label scores, not just top-1
            truncation=True,
            max_length=512,
        )
        print("[FinBERT] Model loaded successfully.")

    def score_batch(self, headlines: List[str]) -> List[Dict[str, float]]:
        """
        Score a list of headlines.

        Returns
        -------
        list of dict
            Each dict: {"positive": float, "neutral": float, "negative": float}
            Values are probabilities summing to 1.
        """
        if not headlines:
            return []

        results = []
        for i in range(0, len(headlines), _DEFAULT_BATCH_SIZE):
            batch = headlines[i : i + _DEFAULT_BATCH_SIZE]
            raw = self._pipe(batch)
            for item in raw:
                # item is a list of {"label": str, "score": float}
                scores = {d["label"].lower(): d["score"] for d in item}
                results.append({
                    "positive": scores.get("positive", 0.0),
                    "neutral":  scores.get("neutral",  0.0),
                    "negative": scores.get("negative", 0.0),
                })
        return results

    def daily_stress(self, headlines: List[str]) -> float:
        """
        Aggregate headlines into a daily stress score in [0, 1].

        Stress = fraction of headlines classified as net-negative
        (negative probability > positive probability).

        Returns np.nan if no headlines provided.
        """
        if not headlines:
            return np.nan

        batch_results = self.score_batch(headlines)
        n_negative = sum(
            1 for r in batch_results if r["negative"] > r["positive"]
        )
        return n_negative / len(batch_results)


# ══════════════════════════════════════════════════════════════════════════════
# FinGPT  (for ≥8 GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════

# Map FinGPT output labels to numeric scores in [-1, +1]
_SENTIMENT_SCORES: Dict[str, float] = {
    "strong negative":     -1.00,
    "moderately negative": -0.67,
    "mildly negative":     -0.33,
    "neutral":              0.00,
    "mildly positive":     +0.33,
    "moderately positive": +0.67,
    "strong positive":     +1.00,
}

_FINGPT_PROMPT = (
    "Instruction: What is the sentiment of this news? Please choose an answer from "
    "{strong negative/moderately negative/mildly negative/neutral/mildly positive/"
    "moderately positive/strong positive}. "
    "Evaluate based only on what is stated in the headline itself; "
    "do not use knowledge of events that occurred after its publication date.\n"
    "Input: {headline}\n"
    "Answer: "
)


class FinGPTSentimentEngine:
    """
    Sentiment scoring using LLaMA2-7B + FinGPT LoRA (~4–5 GB in 4-bit).

    Requires ≥8 GB VRAM. For 4 GB GPUs use FinBERTSentimentEngine instead.

    Parameters
    ----------
    base_model : str
        HuggingFace base LLM model ID.
    lora_model : str
        HuggingFace FinGPT LoRA adapter ID.
    device : str
        "cuda" or "cpu".
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

        print(f"[FinGPT] Loading {self.base_model} + {self.lora_model} on {self.device}")
        print("[FinGPT] Requires ≥8 GB VRAM. For 4 GB GPUs use --model-type finbert.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model = PeftModel.from_pretrained(base, self.lora_model)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        print("[FinGPT] Model loaded successfully.")

    def score_batch(self, headlines: List[str]) -> List[Dict[str, float]]:
        import torch

        prompts = [_FINGPT_PROMPT.format(headline=h) for h in headlines]
        results = []
        batch_size = int(os.environ.get("FINGPT_BATCH_SIZE", "8"))

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                decoded = self._tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                ).strip().lower()
                label = self._parse_label(decoded)
                score = _SENTIMENT_SCORES[label]
                results.append({
                    "positive": max(0.0, score),
                    "neutral":  1.0 - abs(score),
                    "negative": max(0.0, -score),
                })
        return results

    def daily_stress(self, headlines: List[str]) -> float:
        if not headlines:
            return np.nan
        batch_results = self.score_batch(headlines)
        n_negative = sum(1 for r in batch_results if r["negative"] > r["positive"])
        return n_negative / len(batch_results)

    @staticmethod
    def _parse_label(decoded: str) -> str:
        for label in sorted(_SENTIMENT_SCORES, key=len, reverse=True):
            if label in decoded:
                return label
        if "negative" in decoded:
            return "moderately negative"
        if "positive" in decoded:
            return "moderately positive"
        return "neutral"
