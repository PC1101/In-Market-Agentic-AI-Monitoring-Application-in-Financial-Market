# Live Trading Backtest Module

Simulates a live trader starting with $1M capital running a **long-only** strategy
on the S&P 500 universe with the **agentic monitoring layer** as a risk overlay.

## Quick Start (JT Momentum)

```bash
# Step 1: Generate the long-only JT curve (from XSectional/)
cd XSectional && python run_daily_pnl.py --long-only

# Step 2: Run the live backtest (from monitoring/)
cd monitoring
python run_live_backtest.py \
  --curve ../XSectional/results/equity_curve_daily_long_only.csv \
  --strategy JT_MOM \
  --logs-glob "results/agentic_*_JT_MOM.jsonl" \
  --capital 1000000 --tcost-bps 10 --benchmark SPY
```

## AL Stat Arb Plug-In Recipe

To run the same backtest for AL PCA stat arb:

1. **Re-run the stat arb backtest** through 2025 with long-only:
   ```bash
   cd "Stat Arb/statsArb-dev"
   python run_full_universe.py --start 2004-01-02 --end 2025-12-30 --pit --long-only --tag long_only_full
   ```
   This produces `results/full_universe/long_only_full/equity_curve.csv`.

2. **Generate agentic logs** for all 12 windows on the new curve:
   ```bash
   cd monitoring
   for w in quant_meltdown_2007 gfc_lehman_2008 momentum_crash_2009 downgrade_2011 \
            calm_2004_2006 calm_2013_2014 flash_crash_2010 china_deval_2015 \
            volmageddon_2018 covid_2020 calm_2012 calm_2017; do
     python run_agentic.py --window $w --strategy AL_PCA \
       --model ollama:qwen2.5:3b \
       --curve "../Stat Arb/statsArb-dev/results/full_universe/long_only_full/equity_curve.csv"
   done
   ```

3. **Run the live backtest** (same CLI, different args):
   ```bash
   python run_live_backtest.py \
     --curve "../Stat Arb/statsArb-dev/results/full_universe/long_only_full/equity_curve.csv" \
     --strategy AL_PCA \
     --logs-glob "results/agentic_*_AL_PCA.jsonl" \
     --capital 1000000 --tcost-bps 10 --benchmark SPY
   ```

## Exposure Policy

The agentic risk overlay maps supervisor decisions to portfolio exposure using a
graded ladder:

| State    | Action    | Exposure |
|----------|-----------|----------|
| NORMAL   | any       | 100%     |
| WATCH    | any       | 100%     |
| ALERT    | < REDUCE  | 50%      |
| ALERT    | REDUCE    | 25%      |
| CRITICAL | any       | 0%       |
| any      | HALT      | 0%       |

After a de-risk trigger:
- **Cooldown**: minimum exposure held for 5 trading days
- **Staged re-entry**: 3 consecutive clean days per step (50% → 100%)
- **Lag**: decisions at close t apply from t+1 (causal)
- **Uncovered days**: 100% exposure (hybrid coverage assumption)

## Known Limitations

1. **Hybrid coverage bias**: agentic decisions exist only in 12 evaluation windows;
   100% exposure assumed elsewhere.
2. **Curve mismatch**: logs were generated monitoring the long-short curve; applied
   to long-only (regime breaks are market-wide, but it's an approximation).
3. **False-positive drag**: calm windows have ALERT days that cost returns.
4. Strategy curve is frictionless; only overlay trades are costed.
5. Outperformance vs S&P 500 is an empirical outcome, not guaranteed.
