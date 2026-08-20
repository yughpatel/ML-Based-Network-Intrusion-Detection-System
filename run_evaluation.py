"""Entry point: runs the full NIDS model evaluation pipeline (binary,
5-category, and full multiclass KNN classifiers, each tuned via stratified
CV on KDDTrain+ and evaluated on the official held-out KDDTest+ set) and
writes tables to results/tables/ and plots to results/plots/.

Usage: python run_evaluation.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_all

LOG_PATH = Path(__file__).resolve().parent / "results" / "evaluation_log.txt"


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")

    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    t0 = time.time()
    run_all(log=log)
    log(f"\nTotal runtime: {time.time() - t0:.1f}s")
    log_file.close()


if __name__ == "__main__":
    main()
