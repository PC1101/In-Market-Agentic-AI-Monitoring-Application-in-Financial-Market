"""News filtering pipeline (Week 3).

regex/keyword filter -> quantitative signal aggregator -> triage logic, feeding
the News Context Agent and, through it, the Performance Supervisor. All stages
are causal: records are cut off at T-1 relative to the decision date before
anything downstream sees them (see ``pipeline.build_news_block``).
"""

from .records import NewsRecord, load_jsonl_news, load_parquet_news
from .filters import STRESS_LEXICON, classify_headline, split_records
from .sentiment import FakeScorer, FinBERTScorer
from .aggregate import daily_signal
from .triage import TriageMode, triage
from .pipeline import NewsConfig, build_news_block

__all__ = [
    "NewsRecord",
    "load_jsonl_news",
    "load_parquet_news",
    "STRESS_LEXICON",
    "classify_headline",
    "split_records",
    "FakeScorer",
    "FinBERTScorer",
    "daily_signal",
    "TriageMode",
    "triage",
    "NewsConfig",
    "build_news_block",
]
