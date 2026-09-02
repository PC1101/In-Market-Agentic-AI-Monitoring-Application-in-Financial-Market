# China quant literature corpus

Open-access papers on Chinese-market quantitative investing, crowding/regime breaks,
ML asset pricing, HFT/microstructure, stat-arb, and news/LLM-for-finance.
Assembled as a seed / citation / RAG layer for the monitoring project (not a
fine-tune training set — see note at bottom).

`INDEX_full_corpus.csv` is the full 30-paper database (incl. paywalled entries).
The 14 PDFs below are the open-access subset that downloaded cleanly.

## Downloaded PDFs

| File | Paper | Why it's here |
|---|---|---|
| `khandani_lo_quants_aug2007.pdf` | Khandani & Lo, *What Happened to the Quants in August 2007?* | CORE — anatomy of a crowded-quant deleveraging cascade (the event class the monitor targets) |
| `gu_kelly_xiu_empirical_ap_ml.pdf` | Gu, Kelly & Xiu, *Empirical Asset Pricing via Machine Learning* | Benchmark ML asset-pricing method the China papers replicate |
| `arxiv_1904.00745_deep_learning_asset_pricing.pdf` | Chen, Pelger & Zhu, *Deep Learning in Asset Pricing* | No-arbitrage deep net for the SDF |
| `arxiv_2210.12462_deep_multifactor_china.pdf` | *Factor Investing with a Deep Multi-Factor Model* (China) | Deep multi-factor model w/ industry+market neutralization |
| `arxiv_2112.03170_ff5_vs_ff3_china.pdf` | *FF5 vs FF3 on China A-share* | Factor structure on 2005-2020 A-shares |
| `arxiv_2512.16251_consensus_bottleneck_ap.pdf` | *Interpretable DL for Stock Returns (Consensus-Bottleneck)* | Interpretable pricing — ties to reasoning-quality rubric |
| `arxiv_2501.03171_hf_leadlag_china_futures.pdf` | *HF lead-lag in Chinese stock index futures* | Tick-level price discovery, CFFEX calendar spreads |
| `arxiv_1710.07470_hf_technical_rules_china.pdf` | *Stationary technical trading rules, Chinese index futures* | HF technical-rule profitability |
| `arxiv_2406.10695_statarb_graph_clustering.pdf` | *Stat-arb multi-pair via graph clustering* | Graph-clustering stat-arb (A-share test window) |
| `arxiv_2403.12180_statarb_rl.pdf` | *Advanced Statistical Arbitrage with RL* | RL stat-arb methodology |
| `arxiv_2509.23609_llm_futures_factors_china.pdf` | *LLMs and Futures Price Factors in China* | CORE — closest analogue to the news-agent + Qwen fine-tune plan |
| `arxiv_2407.16150_finbert_lstm_news.pdf` | *FinBERT-LSTM: news sentiment for stock prediction* | News-sentiment -> price pipeline |
| `arxiv_2306.02136_finbert_sentiment.pdf` | *Financial sentiment analysis using FinBERT* | FinBERT baseline for the news-context agent |
| `arxiv_2510.06864_news_topics_stock_movement.pdf` | *Framework for measuring how news topics drive stock movement* | News-topic -> return attribution (supervisor root-cause) |

## Not downloaded

- **CAIA** *Factor Investing in the China A-Share Market* — Cloudflare-blocked; download
  manually from https://caia.org/sites/default/files/factor_investing_in_the_china_a-share_market.pdf
- **Paywalled** (ScienceDirect / Springer / Wiley / SSRN-restricted) — see `INDEX_full_corpus.csv`
  rows with `open_access = No`. Need institutional access for full text.

## Note on fine-tuning

This corpus is ~14 MB of text — appropriate as a **RAG / few-shot grounding / citation**
layer, NOT as a standalone fine-tune set for a 27B model (which wants hundreds of MB-GBs).
For domain adaptation, pair these as a seed layer on top of the FNSPID news corpus.
