# Methodology corpus — the literature your SYSTEM is built on

Third scraping pass, keyed to the project's own machinery rather than the China-fund
angle: change-point/regime detection (classical layer), LLM agents for finance
(agentic layer), and crowding/fire-sale dynamics (the events you detect).
`INDEX_by_topic.csv` has full metadata + per-paper relevance.

## The must-reads (flagged CORE in the CSV)

| File | Why it matters to THIS project |
|---|---|
| `changepoint__arxiv_0710.3742_adams_mackay_bocpd.pdf` | **The BOCPD algorithm your `monitoring/detectors/` BOCPD is a direct implementation of.** Primary methods citation. |
| `llm_agent__arxiv_2510.07920_profit_mirage_info_leakage.pdf` | **Information leakage in LLM financial agents** — the exact lookahead/training-leak failure your information-parity guardrails + two-pass control are designed to prevent. Cite in your methodology spine. |
| `llm_agent__arxiv_2412.20138_tradingagents_multiagent.pdf` | Multi-agent LLM trading firm (analyst/trader roles) — closest published architecture to your news-agent → supervisor design. |

## Contents (12 open-access PDFs)

- **Change-point / regime detection (3):** Adams-MacKay BOCPD, BOCPD on order flow, regime detection via realized covariances
- **Early warning (1):** review of crisis early-warning systems
- **LLM agents for finance (6):** TradingAgents, Profit Mirage (info leakage), FinSphere, FinVision, MountainLion, Agentic-LLM survey
- **Crowding / systemic risk (2):** Suffocating Fire Sales, ECB crowding measurement

## Note

Same as the rest of the corpus: PDFs are gitignored (public repo). Run
`../download.sh` to (re)populate all three folders from source.
