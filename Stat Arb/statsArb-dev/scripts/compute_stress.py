#!/usr/bin/env python3
"""
Run FinBERT/FinGPT inference on FNSPID headlines to produce a regime stress series.

Requires (install first):
    pip install torch transformers huggingface_hub yfinance

For FinGPT (--model-type fingpt, requires >=8 GB VRAM):
    pip install peft accelerate bitsandbytes

Steps:
  1. Load sector-filtered FNSPID headlines (run cache_news.py first).
  2. Load sentiment model on GPU.
     - finbert (default): ProsusAI/finbert (~400 MB, fits 4 GB GPU)
     - fingpt: LLaMA2-7B + LoRA (~14 GB download, requires >=8 GB VRAM)
  3. Score each trading day's headlines → daily stress in [0, 1].
     Stress on day T is derived from T-1 headlines (look-ahead guard).
  4. Save stress series: data/news/stress_<etf>.csv
  5. Generate diagnostic chart: results/fingpt_stress_vs_vix_<etf>.png
     (compare stress to VIX to check for training-data bias)

LOOK-AHEAD BIAS WARNING:
  Both models were trained on financial corpora that may include post-crisis
  retrospectives. Inspect the diagnostic chart carefully before trusting
  backtest results from crisis periods (GFC 2008, COVID 2020).

Usage:
    python scripts/compute_stress.py --etf xlf
    python scripts/compute_stress.py --etf xlf --model-type fingpt  # >=8 GB VRAM only
    python scripts/compute_stress.py --etf xlf --device cpu         # slow, for testing
    python scripts/compute_stress.py --etf xlk --threshold 0.35
"""
import argparse
import os
import sys
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.news_loader import FNSPIDLoader
from src.fingpt_sentiment import (
    FinBERTSentimentEngine,
    FinGPTSentimentEngine,
    FINBERT_MODEL_DEFAULT,
    FINGPT_BASE_DEFAULT,
    FINGPT_LORA_DEFAULT,
)
from src.regime_filter import RegimeFilter


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--etf", required=True,
                   help="Sector ETF ticker (e.g. xlf).")
    p.add_argument("--model-type", default="finbert", choices=["finbert", "fingpt"],
                   help="Sentiment backend: finbert (default, <=8 GB GPU) or fingpt (>=8 GB VRAM).")
    p.add_argument("--finbert-model", default=FINBERT_MODEL_DEFAULT,
                   help=f"HuggingFace FinBERT model (default: {FINBERT_MODEL_DEFAULT}).")
    p.add_argument("--base-model", default=FINGPT_BASE_DEFAULT,
                   help=f"[fingpt only] HuggingFace base LLM (default: {FINGPT_BASE_DEFAULT}).")
    p.add_argument("--lora-model", default=FINGPT_LORA_DEFAULT,
                   help=f"[fingpt only] HuggingFace LoRA adapter (default: {FINGPT_LORA_DEFAULT}).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Inference device (default: cuda).")
    p.add_argument("--cache-dir", default="data/news",
                   help="Directory with FNSPID sector cache (default: data/news).")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Stress threshold to annotate on chart (default: 0.40).")
    return p.parse_args()


def _load_bt_dates(cfg: dict, etf: str) -> pd.DatetimeIndex:
    """Return the full DatetimeIndex of trading days from the price CSV."""
    prices_path = cfg["prices_file_path"].get(etf)
    if not prices_path:
        raise ValueError(f"ETF '{etf}' not in config prices_file_path.")
    df = pd.read_csv(prices_path, index_col="Date", parse_dates=True).sort_index()
    return df.index


def _generate_diagnostic_chart(
    stress: pd.Series, etf: str, threshold: float, out_path: str
):
    """
    Plot FinGPT stress alongside VIX for bias inspection.

    A well-calibrated signal should spike during known crises (2020-03,
    2018-Q4, 2015-08) and be calm during bull markets.
    """
    try:
        import yfinance as yf
        vix_raw = yf.download(
            "^VIX",
            start=str(stress.index.min().date()),
            end=str(stress.index.max().date()),
            progress=False,
        )
        vix = vix_raw["Close"].squeeze()
        has_vix = not vix.empty
    except Exception:
        warnings.warn("[compute_stress] Could not download VIX.")
        has_vix = False

    n_panels = 2 if has_vix else 1
    fig, axs = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axs = [axs]

    axs[0].fill_between(stress.index, stress.fillna(0).values,
                        alpha=0.4, color="tomato", label="Sentiment stress (FNSPID)")
    axs[0].plot(stress.index, stress.values, lw=0.5, color="firebrick", alpha=0.7)
    axs[0].axhline(threshold, ls="--", lw=1.2, color="darkred",
                   label=f"Threshold ({threshold:.0%})")
    axs[0].set_ylabel("Stress\n(fraction negative headlines)")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(loc="upper left", fontsize=9)
    axs[0].set_title(
        f"Regime Stress — {etf.upper()} (source: FNSPID)\n"
        "WARNING: Model training data may include post-crisis retrospectives. "
        "Compare against VIX to assess look-ahead bias before trusting results."
    )

    if has_vix:
        axs[1].plot(vix.index, vix.values, lw=1, color="steelblue", label="VIX")
        axs[1].axhline(30, ls="--", lw=0.8, color="navy", alpha=0.6,
                       label="VIX=30 (elevated)")
        axs[1].set_ylabel("VIX")
        axs[1].set_ylim(bottom=0)
        axs[1].legend(loc="upper left", fontsize=9)
        axs[1].set_title(
            "VIX (contemporaneous reference — no GPT look-ahead bias)"
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compute_stress] Diagnostic chart → {out_path}")


def main():
    args = parse_args()
    etf = args.etf.lower()

    with open("configs/optimise_trading_rules.yml") as fh:
        cfg = yaml.load(fh, Loader=yaml.SafeLoader)

    default_start, default_end = cfg.get("bt_dt", {}).get(
        etf, (cfg["st_dt"], cfg["ed_dt"])
    )

    model_label = args.model_type.upper()
    print("=" * 60)
    print(f"  ETF        : {etf.upper()}")
    print(f"  Date range : {default_start} -> {default_end}")
    print(f"  Model type : {model_label}")
    print(f"  Device     : {args.device}")
    if args.model_type == "finbert":
        print(f"  FinBERT    : {args.finbert_model}")
    else:
        print(f"  Base model : {args.base_model}")
        print(f"  LoRA model : {args.lora_model}")
    print(f"  Threshold  : {args.threshold:.0%}")
    print("=" * 60)

    # Step 1: Load FNSPID headlines for the sector
    print(f"\n[1/3] Loading FNSPID headlines for {etf.upper()}...")
    loader = FNSPIDLoader(etf=etf, cache_dir=args.cache_dir)
    news_by_date = loader.load_titles_by_date(default_start, default_end)
    n_days = sum(1 for v in news_by_date.values() if v)
    total_headlines = sum(len(v) for v in news_by_date.values())
    print(f"  {n_days} days with headlines ({total_headlines:,} total articles).")

    # Step 2: Load sentiment engine
    print(f"\n[2/3] Loading {model_label} on {args.device}...")
    if args.model_type == "finbert":
        print("  (First run downloads ~400 MB ProsusAI/finbert from HuggingFace)")
        engine = FinBERTSentimentEngine(
            model_name=args.finbert_model,
            device=args.device,
        )
    else:
        print("  (First run downloads ~14 GB LLaMA2 base model from HuggingFace)")
        print("  WARNING: FinGPT requires >=8 GB VRAM. Use --model-type finbert for 4 GB GPUs.")
        engine = FinGPTSentimentEngine(
            base_model=args.base_model,
            lora_model=args.lora_model,
            device=args.device,
        )

    # Step 3: Build stress series (T-1 look-ahead guard enforced inside build())
    print("\n[3/3] Scoring headlines and building stress series...")
    print("  (May take hours for multi-year ranges; progress printed per day)")
    bt_dates = _load_bt_dates(cfg, etf)
    bt_dates = bt_dates[
        (bt_dates >= pd.Timestamp(default_start))
        & (bt_dates <= pd.Timestamp(default_end))
    ]

    output_path = os.path.join(args.cache_dir, f"stress_{etf}.csv")
    rf = RegimeFilter()
    stress = rf.build(
        news_by_date=news_by_date,
        model=engine,
        bt_dates=bt_dates,
        output_path=output_path,
    )

    above = (stress > args.threshold).sum()
    print(f"\n=== Stress summary ===")
    print(f"  Mean stress              : {stress.mean():.3f}")
    print(f"  Max stress               : {stress.max():.3f}")
    print(f"  Days above {args.threshold:.0%} threshold  : {above} "
          f"({above / max(stress.notna().sum(), 1):.1%} of days with data)")

    # Diagnostic chart
    diag_path = os.path.join("results", f"stress_vs_vix_{etf}.png")
    _generate_diagnostic_chart(stress, etf, args.threshold, diag_path)

    print(f"\n=== Next steps ===")
    print(f"  1. Inspect {diag_path} for GPT look-ahead bias.")
    print(f"  2. Run with filter:")
    print(f"       python run_backtest.py --etf {etf} --regime-filter {output_path}")
    print(f"  3. Compare vs baseline:")
    print(f"       python run_backtest.py --etf {etf}")


if __name__ == "__main__":
    main()
