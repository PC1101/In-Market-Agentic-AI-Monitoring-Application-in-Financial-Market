#!/usr/bin/env bash
# Re-download the open-access literature corpus from original sources.
# The PDFs themselves are gitignored (copyrighted; this is a public repo),
# so run this to populate data/literature/ locally.
#   usage: bash data/literature/download.sh
set -u
cd "$(dirname "$0")"
mkdir -p by_author
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# filename|url   (topical corpus + author-attributed, all open-access)
LIST=$(cat <<'EOF'
khandani_lo_quants_aug2007.pdf|https://web.mit.edu/Alo/www/Papers/august07b_2.pdf
gu_kelly_xiu_empirical_ap_ml.pdf|http://arc.hhs.se/download.aspx?MediumId=5617
arxiv_2112.03170_ff5_vs_ff3_china.pdf|https://arxiv.org/pdf/2112.03170
arxiv_2210.12462_deep_multifactor_china.pdf|https://arxiv.org/pdf/2210.12462
arxiv_2512.16251_consensus_bottleneck_ap.pdf|https://arxiv.org/pdf/2512.16251
arxiv_1904.00745_deep_learning_asset_pricing.pdf|https://arxiv.org/pdf/1904.00745
arxiv_2501.03171_hf_leadlag_china_futures.pdf|https://arxiv.org/pdf/2501.03171
arxiv_1710.07470_hf_technical_rules_china.pdf|https://arxiv.org/pdf/1710.07470
arxiv_2406.10695_statarb_graph_clustering.pdf|https://arxiv.org/pdf/2406.10695
arxiv_2403.12180_statarb_rl.pdf|https://arxiv.org/pdf/2403.12180
arxiv_2509.23609_llm_futures_factors_china.pdf|https://arxiv.org/pdf/2509.23609
arxiv_2407.16150_finbert_lstm_news.pdf|https://arxiv.org/pdf/2407.16150
arxiv_2306.02136_finbert_sentiment.pdf|https://arxiv.org/pdf/2306.02136
arxiv_2510.06864_news_topics_stock_movement.pdf|https://arxiv.org/pdf/2510.06864
by_author/yuyuan_mingshi__nber_w16898_short_of_it_sentiment_anomalies.pdf|https://www.nber.org/system/files/working_papers/w16898/w16898.pdf
by_author/yuyuan_mingshi__nber_w18560_arbitrage_asymmetry_ivol.pdf|https://www.nber.org/system/files/working_papers/w18560/w18560.pdf
by_author/yuyuan_mingshi__wharton_stambaugh_yuan_mispricing_factors.pdf|https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/05.15Stambaugh1.pdf
by_author/chenyingzhang_egret__nber_w17828_chinas_financial_system.pdf|https://www.nber.org/system/files/working_papers/w17828/w17828.pdf
by_author/penghu_loshu__arxiv_1008.3034_snell_envelope_multiplicative.pdf|https://arxiv.org/pdf/1008.3034
by_author/penghu_loshu__arxiv_1107.1948_concentration_interacting_particle.pdf|https://arxiv.org/pdf/1107.1948
by_author/penghu_loshu__inria_robustness_snell_envelope.pdf|https://people.bordeaux.inria.fr/pierre.delmoral/robustness-snell-particle.pdf
EOF
)

ok=0; fail=0
while IFS='|' read -r fn url; do
  [ -z "$fn" ] && continue
  curl -sL -A "$UA" --max-time 90 --retry 2 -o "$fn" "$url"
  if [ "$(file -b --mime-type "$fn" 2>/dev/null)" = "application/pdf" ]; then
    printf "OK   %s\n" "$fn"; ok=$((ok+1))
  else
    printf "FAIL %s  <-- fetch manually from %s\n" "$fn" "$url"; rm -f "$fn"; fail=$((fail+1))
  fi
done <<< "$LIST"
echo "----"; echo "downloaded=$ok failed=$fail"
echo "See INDEX_full_corpus.csv and by_author/INDEX_by_author.csv for full metadata + paywalled entries."
