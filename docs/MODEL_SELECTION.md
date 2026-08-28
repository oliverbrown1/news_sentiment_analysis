# V2 Sentiment Model Selection

Reviewed: 2026-08-28

## Decision

V2 defaults to `neoyipeng/ModernFinBERT-base`; V1 retains
`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` as the frozen
baseline. On the pinned FinEntity benchmark, V2 reaches 75.41% accuracy and
75.38% macro F1; V1 reaches 69.03% accuracy and 69.58% macro F1.

## Why This Candidate

The model is a 149M-parameter ModernBERT sequence classifier released under
Apache-2.0. Its documented training set combines earnings calls, financial news,
analyst Q&A, tweets, and Financial PhraseBank examples. The model card reports
81.41% five-fold cross-validation accuracy and 77.66% macro F1 across its combined
dataset, plus 80.52% macro F1 on Financial PhraseBank 50Agree when those examples
are excluded from training.

This is more current and broadly trained than the V1 DistilRoBERTa checkpoint,
while remaining small enough for local inference. It also preserves the same
three labels, which makes comparison through `eval` straightforward.

Source: https://huggingface.co/neoyipeng/ModernFinBERT-base

## Caveats

- It is still sequence-level sentiment, not natively entity-targeted. V2 selects
  target evidence before inference to make that limitation explicit.
- The checkpoint requires Transformers 5 and publishes generic label IDs. The V2
  adapter records the documented mapping: `LABEL_0` negative, `LABEL_1` neutral,
  and `LABEL_2` positive.
- The model card says roughly 68% of its news training examples are Canadian
  mining press releases, so domain shift is plausible.
- Published model-card scores are not evidence that it improves this project.
  The pinned FinEntity reports are the evidence used for the local comparison.
- V2 trades throughput for quality: its measured mean was 43.8 ms per annotation
  versus 8.5 ms for V1 on the recorded Apple MPS runs.
- Financial PhraseBank is common in finance-model training, so scores on that
  dataset can overstate generalization unless overlap is excluded.

`tabularisai/ModernFinBERT` was also reviewed. Its model card reports an average
F1 of 0.63 over its selected evaluation sets, versus 0.66 for the older
DistilRoBERTa comparator, so it did not provide a stronger default rationale.

Source: https://huggingface.co/tabularisai/ModernFinBERT
