# NetGuard

NetGuard is a machine learning pipeline that classifies network traffic on the NSL-KDD dataset, built as a university coursework project. It compares K-Nearest Neighbors and class-weighted Random Forest across three label granularities (binary normal/attack, 5-category attack type, and full multiclass), with and without SMOTE oversampling for rare attack classes.

## How it works

- Each network connection record is described by features like duration, protocol type, and byte count.
- Models are tuned via stratified k-fold cross-validation on `KDDTrain+.txt`, then evaluated once on the official held-out `KDDTest+.txt` (which includes attack types never seen during training, by NSL-KDD design).
- Three label granularities are evaluated: `binary` (normal/attack), `category5` (normal/dos/probe/r2l/u2r), and `multiclass` (raw attack type).
- Two model families are compared: KNN and class-weighted Random Forest, each with a SMOTE variant that oversamples rare classes (capped growth, so classes like U2R aren't inflated to the size of "normal").

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved, de-duplicated version of the original KDD Cup 1999 dataset, commonly used as a benchmark for intrusion detection research.

## Tech Stack

- **Language**: Python
- **ML**: scikit-learn (KNN, Random Forest), imbalanced-learn (SMOTE)

## Results (held-out KDDTest+)

See [`RESULTS.md`](RESULTS.md) for a summary of findings across models and granularities. Full numbers are in `results/tables/summary_all_granularities_testset.csv` (accuracy, weighted/macro precision-recall-F1, best CV params) and `results/tables/*_per_class_report_testset.csv` (per-class breakdowns). `results/plots/` has confusion matrices per combination.

## Status

This model is also reused in a second project, [TraceBack](https://github.com/yughpatel/TraceBack), applying the same KNN approach to server log analysis instead of network traffic.

Currently being refined based on faculty feedback:
- [x] Add confusion matrix and full evaluation metrics (accuracy, precision, recall, F1-score)
- [x] Improve feature scaling before distance-based classification
- [x] Address class imbalance (class-weighted Random Forest, SMOTE oversampling)
- [x] Try a gradient-boosted tree model as a third model family
- [ ] Retrain on a larger / more modern dataset (e.g. CIC-IDS2017/2018)
- [ ] Package a saved model + inference script for scoring new records

`notebooks/analysis.ipynb` is an earlier, simpler exploratory pass (single KNN model, single train/test split) kept for reference; `src/` is the actively maintained pipeline described above.

## Setup

```
git clone https://github.com/yughpatel/ML-Based-Network-Intrusion-Detection-System.git
cd ML-Based-Network-Intrusion-Detection-System
pip install -r requirements.txt
python run_evaluation.py
```
