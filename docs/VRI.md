**Week 1: Data \+ Model**

- Point-in-time S\&P 500 universe builder working for any date 2007-2016, with delisted-name handling documented  
- AL PCA baseline re-run on point-in-time data, with August 2007 quant meltdown visible in daily equity curve (gating test)  
- JT baseline re-run on point-in-time data, with April 2009 momentum crash visible (gating test)  
- Daily PnL output schema locked for both strategies (agreed columns, frequency, attribution fields)  
- Local AI model selected, benchmarked, and confirmed running locally with adequate latency for the agentic layer

**Week 2: Classical monitoring \+ agentic structure** 

- Page-Hinkley, BOCPD, HMM, and distributional threshold detectors implemented and unit-tested  
- Aggregation rule (≥2 detectors firing within 5 days) implemented  
- Classical monitoring run end-to-end on both strategies across all six windows (4 event, 2 calm)  
- Per-detector and aggregated metrics computed: detection latency, false positive rate, precision, recall  
- Agentic framework scaffolded: prompt templates, output schemas (structured JSON with state, action, root-cause, confidence), local model integration, logging  
- Information-parity guardrails implemented at the framework level (as-of dating, time-stamp filtering)

**Week 3: News pipeline \+ agent build**

- FNSPID ingested and indexed by date, ready to query for any date in the backtest window  
- Macro data sources (ALFRED vintage, FRED, BLS/Treasury) integrated if in scope  
- News filtering pipeline working: regex/keyword filter → quantitative signal aggregator → triage logic (cheap-model / thinking-model / classical-escalation modes)  
- News Context Agent v1: ingests filtered news, outputs structured risk flags and narrative  
- Performance Supervisor Agent v1: ingests telemetry \+ classical outputs \+ news summary, outputs structured JSON assessment  
- End-to-end agentic run on a single event window (Aug 2007\) succeeding with valid JSON output

**Week 4: Agent refinement \+ full pipeline integration**

- Ensure the model built is bias-proof, demonstrate how it is validated and integrated as part of the agent.   
- Agent prompts iterated and version-controlled  
- Two-pass evaluation mode implemented (with and without events in LLM training data) for hindsight-bias control  
- Agentic system runs cleanly across all six windows for both strategies without manual intervention  
- Per-event sanity checks: agentic outputs reviewed manually on at least one event window to confirm reasoning quality and JSON validity  
- Qualitative scoring rubric drafted for blind reasoning quality assessment

**Week 5: Evaluation \+ results** 

- Full 2x2 results: both strategies × both monitoring conditions × all six windows  
- Headline metrics computed: detection latency, false positive rate, precision, recall  
- Two-pass training-leak control completed; results reported both ways  
- Qualitative reasoning scoring completed using the Week 4 rubric (blind if possible, with both project members scoring each other's outputs) 	  
- Hypothesis results stated explicitly: H1 (faster \+ lower FPR), H2 (generalises across strategies), H3 (advantage differs by strategy) — supported, partially supported, or rejected with evidence

**Week 6: Writing**

- Methodology section complete (data, strategies, classical monitoring, agentic monitoring, evaluation design)  
- Results section complete (tables, figures, hypothesis verdicts)  
- Discussion section complete (H1/H2/H3 interpretation, limitations, future work)  
- Introduction, abstract, and conclusion written  
- Final supervisor read incorporated  
- Do whatever’s left (buffer time)