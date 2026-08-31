# Results Summary

Full numbers: `results/tables/summary_all_granularities_testset.csv` (aggregate) and
`results/tables/*_per_class_report_testset.csv` (per-class). Confusion matrices are in
`results/plots/`. All figures below are on the held-out official `KDDTest+` set, which
includes attack types never seen during training.

## Best model per granularity (by macro-F1)

| granularity | best model | accuracy | f1_macro | vs. majority-class baseline (acc) |
|---|---|---|---|---|
| binary | gradient_boosting | 0.794 | 0.793 | 0.431 |
| category5 | gradient_boosting_smote | 0.782 | 0.644 | 0.431 |
| multiclass | gradient_boosting | 0.722 | 0.456 | 0.431 |

## Observations

- **Gradient boosting (`HistGradientBoostingClassifier`, class-weighted) is the strongest
  model family overall**, beating both KNN and Random Forest on every granularity. The
  gap widens as the label space gets harder: on `category5` it beats RF's macro-F1 by
  ~0.10-0.13, and on `multiclass` by ~0.04.
- **SMOTE helps KNN's rare-class recall the most** (e.g. `category5` KNN macro-F1 0.543
  -> 0.567 with SMOTE) because KNN has no other mechanism for weighting rare classes.
  It has little to no effect on Random Forest and gradient boosting, since both already
  use `class_weight="balanced"` to cover similar ground -- and for `multiclass`
  gradient boosting, SMOTE actually hurts slightly (f1_macro 0.456 -> 0.418), likely
  because interpolating synthetic points between a handful of real examples (some
  multiclass attack types have under 10 training rows) adds noise rather than signal.
- **U2R and R2L remain the hardest classes across every model** -- they're the rarest
  in training and the most behaviorally similar to normal traffic, so even the best
  per-class recall for these classes stays well below what's achieved on DoS/Probe/
  normal. See the per-class reports for exact numbers per model.
- **Multiclass generalization to unseen attacks is the ceiling on performance here**: all
  models plateau around 0.70-0.72 accuracy on `multiclass`, well below `binary`'s ~0.79,
  because KDDTest+ contains attack types absent from KDDTrain+ by design (an intentional
  NSL-KDD generalization test, not a modeling gap that more tuning would close).

## Open next steps

- Retrain/evaluate on a more modern dataset (e.g. CIC-IDS2017/2018) to see whether these
  same relative rankings (gradient boosting > RF > KNN, SMOTE mainly helping KNN) hold.
- Package a saved model + inference script for scoring a single new record, rather than
  evaluation-only.
