# Evaluation Data

`finentity.json` is the FinEntity entity-level financial sentiment dataset. It
contains 979 financial-news paragraphs and 2,131 positive, neutral, or negative
entity annotations.

- Source: https://github.com/yixuantt/FinEntity
- Revision: `3b6cedc5485b669c2ed168f1d949f517636eb7b8`
- SHA-256: `3208667de69383120b0380aebeaabe360669eda72269c33e1c8b09d63df55463`
- Licence: Open Data Commons Attribution License (ODC-By)
- Paper: https://aclanthology.org/2023.emnlp-main.956/

The file is retained unchanged. The evaluator reports known span mismatches and
duplicate annotations rather than silently rewriting the source data.

`finmarba.csv` is the released FinMarBa subset for ticker-level market-direction
evaluation. It contains 8,142 source rows from 2010-01-04 to 2011-12-30 and
produces 9,978 labelled ticker examples; the full corpus described by the paper
is not publicly included in this file.

- Source: https://huggingface.co/datasets/baptle/financial_headlines_market_based
- Revision: `d4ac449b57cb7dddfd5d8e05fd1326f8cb03fa29`
- SHA-256: `554afe421d62f6a93f342ba10def593a485120c86d5e5074477d03dd95258a8c`
- Licence: MIT
- Paper: https://arxiv.org/abs/2507.22932

The CSV is retained unchanged. `market_eval` reports ticker mentions without a
corresponding label and excludes only those unlabelled mentions from evaluation.

FinMarBa has limitation that 2012+ data is not publicly available, while NewsAPI goes back 5 years only.

**Workaround:** Custom dataset created using NewsAPI, seed using `uv run market-eval-build`, which generates `recent_market_headlines.jsonl` using `market_eval_seed.json`.