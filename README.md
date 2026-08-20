# NetGuard

NetGuard is a machine learning model that classifies network traffic as **Normal** or **Attack**, built as a university coursework project. It uses a K-Nearest Neighbors (KNN) algorithm trained on the NSL-KDD dataset.

## How it works

- Each network connection record is described by features like duration, protocol type, and byte count.
- A KNN classifier compares new traffic records against the training set and labels them Normal or Attack based on the closest matching records.
- The goal is to detect intrusion patterns from traffic behavior rather than relying on fixed rules or signatures.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved, de-duplicated version of the original KDD Cup 1999 dataset, commonly used as a benchmark for intrusion detection research.

## Tech Stack

- **Language**: Python
- **ML**: Scikit-learn (KNN)

## Status

This model is also reused in a second project, [TraceBack](https://github.com/yughpatel/TraceBack), applying the same KNN approach to server log analysis instead of network traffic.

Currently being refined based on faculty feedback:
- [ ] Add confusion matrix and full evaluation metrics (accuracy, precision, recall, F1-score)
- [ ] Retrain on a larger / more modern dataset (e.g. CIC-IDS2017/2018)
- [ ] Improve feature scaling before distance-based classification

## Setup

```
git clone https://github.com/yughpatel/ML-Based-Network-Intrusion-Detection-System.git
cd ML-Based-Network-Intrusion-Detection-System
pip install -r requirements.txt
```
