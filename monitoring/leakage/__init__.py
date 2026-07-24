"""Pretraining-leakage control harness (preregistration §8).

Three conditions run on every test event window:
  A — Standard frozen pipeline as-is.
  B — Date-masked: all ISO dates replaced with XXXX-XX-XX before LLM sees context.
  C — Synthetic: M=10 block-bootstrapped calm windows with injected crash templates.

Leakage bound:
  evidence_skill >= performance_C
  memorisation   <= performance_A - min(performance_B, performance_C)
"""

from .synthetic import generate_synthetic_windows, SyntheticWindow

__all__ = ["generate_synthetic_windows", "SyntheticWindow"]
