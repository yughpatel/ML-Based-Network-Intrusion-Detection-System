import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src.constants import RANDOM_STATE
from src.data_loader import load_test, load_train
from src.evaluate import (
    majority_baseline_accuracy,
    per_class_report,
    save_confusion_matrix_plot,
    save_table,
    summary_metrics,
)
from src.preprocessing import Preprocessor
from src.targets import TARGET_BUILDERS

K_GRID = [1, 3, 5, 7, 11, 15, 21]
MAX_CV_FOLDS = 5


def _cv_folds_for(y) -> int:
    min_class_count = pd.Series(y).value_counts().min()
    return max(2, min(MAX_CV_FOLDS, int(min_class_count)))


def select_k_via_cv(X, y, k_grid=K_GRID, log=print) -> pd.DataFrame:
    n_splits = _cv_folds_for(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = np.asarray(X)
    y = np.asarray(y)

    rows = []
    for k in k_grid:
        fold_metrics = []
        for train_idx, val_idx in skf.split(X, y):
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn.fit(X[train_idx], y[train_idx])
            pred = knn.predict(X[val_idx])
            fold_metrics.append(summary_metrics(y[val_idx], pred))
        mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        std_acc = pd.DataFrame(fold_metrics)["accuracy"].std()
        row = {"k": k, "cv_folds": n_splits, "acc_std": std_acc, **mean_metrics}
        rows.append(row)
        log(f"    k={k:>3}  acc={row['accuracy']:.4f}  f1_macro={row['f1_macro']:.4f}  f1_weighted={row['f1_weighted']:.4f}")
    return pd.DataFrame(rows)


def run_granularity(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, pre: Preprocessor, log=print) -> dict:
    log(f"\n=== {name} ===")
    y_train, y_test = TARGET_BUILDERS[name](train_df, test_df)
    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)

    log(f"  train class counts:\n{pd.Series(y_train).value_counts().to_string()}")

    log("  selecting k via stratified CV on the training split...")
    t0 = time.time()
    cv_results = select_k_via_cv(X_train, y_train, log=log)
    log(f"  CV took {time.time() - t0:.1f}s")
    save_table(cv_results, f"{name}_cv_k_selection.csv")

    best_row = cv_results.loc[cv_results["f1_macro"].idxmax()]
    best_k = int(best_row["k"])
    log(f"  best k = {best_k} (by CV macro-F1 = {best_row['f1_macro']:.4f})")

    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    test_metrics = summary_metrics(y_test, y_pred)
    baseline_acc = majority_baseline_accuracy(y_train, y_test)
    test_metrics["majority_class_baseline_accuracy"] = baseline_acc
    test_metrics["best_k"] = best_k
    test_metrics["granularity"] = name
    log(f"  held-out KDDTest+ : accuracy={test_metrics['accuracy']:.4f}  "
        f"f1_macro={test_metrics['f1_macro']:.4f}  f1_weighted={test_metrics['f1_weighted']:.4f}  "
        f"(majority-class baseline accuracy={baseline_acc:.4f})")

    labels = sorted(set(y_train) | set(y_test))
    report = per_class_report(y_test, y_pred, labels=labels)
    save_table(report, f"{name}_per_class_report_testset.csv")
    save_confusion_matrix_plot(
        y_test, y_pred, labels,
        title=f"{name} — confusion matrix on KDDTest+ (k={best_k})",
        filename=f"{name}_confusion_matrix_testset.png",
    )

    return {
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "per_class_report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_all(log=print) -> dict:
    log("Loading data...")
    train_df = load_train()
    test_df = load_test()
    log(f"  train: {train_df.shape}, test: {test_df.shape}")

    log("Fitting preprocessor on training split only...")
    pre = Preprocessor().fit(train_df)

    results = {}
    for name in ["binary", "category5", "multiclass"]:
        results[name] = run_granularity(name, train_df, test_df, pre, log=log)

    summary = pd.DataFrame([results[name]["test_metrics"] for name in results])
    cols = ["granularity", "best_k", "accuracy", "majority_class_baseline_accuracy",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro"]
    summary = summary[cols]
    save_table(summary, "summary_all_granularities_testset.csv")
    log("\n=== Summary (held-out KDDTest+) ===")
    log(summary.to_string(index=False))

    return results


if __name__ == "__main__":
    run_all()
