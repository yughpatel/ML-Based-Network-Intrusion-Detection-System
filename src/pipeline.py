import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

MAX_CV_FOLDS = 5

# Each model is a (param -> estimator) factory plus the grid of param dicts to
# try via CV. Random forest uses class_weight="balanced" so rare classes
# (R2L, U2R) contribute as much to the split criterion as "normal"/"dos" do,
# instead of being swamped -- KNN has no such knob, which is why it misses
# almost all R2L/U2R on category5/multiclass (see per-class reports).
MODEL_REGISTRY = {
    "knn": {
        "factory": lambda n_neighbors: KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        "param_grid": [{"n_neighbors": k} for k in [1, 3, 5, 7, 11, 15, 21]],
    },
    "random_forest": {
        "factory": lambda n_estimators, max_depth: RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "param_grid": [
            {"n_estimators": 200, "max_depth": None},
            {"n_estimators": 200, "max_depth": 20},
            {"n_estimators": 300, "max_depth": None},
            {"n_estimators": 500, "max_depth": None},
        ],
    },
}


def _cv_folds_for(y) -> int:
    min_class_count = pd.Series(y).value_counts().min()
    return max(2, min(MAX_CV_FOLDS, int(min_class_count)))


def select_hyperparams_via_cv(model_key: str, X, y, log=print) -> pd.DataFrame:
    factory = MODEL_REGISTRY[model_key]["factory"]
    param_grid = MODEL_REGISTRY[model_key]["param_grid"]
    n_splits = _cv_folds_for(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = np.asarray(X)
    y = np.asarray(y)

    rows = []
    for params in param_grid:
        fold_metrics = []
        for train_idx, val_idx in skf.split(X, y):
            model = factory(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            fold_metrics.append(summary_metrics(y[val_idx], pred))
        mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        std_acc = pd.DataFrame(fold_metrics)["accuracy"].std()
        row = {**params, "cv_folds": n_splits, "acc_std": std_acc, **mean_metrics}
        rows.append(row)
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        log(f"    {params_str:<28}  acc={row['accuracy']:.4f}  f1_macro={row['f1_macro']:.4f}  f1_weighted={row['f1_weighted']:.4f}")
    return pd.DataFrame(rows)


def run_granularity(name: str, model_key: str, train_df: pd.DataFrame, test_df: pd.DataFrame, pre: Preprocessor, log=print) -> dict:
    log(f"\n=== {name} / {model_key} ===")
    y_train, y_test = TARGET_BUILDERS[name](train_df, test_df)
    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)

    log(f"  train class counts:\n{pd.Series(y_train).value_counts().to_string()}")

    log(f"  selecting hyperparams via stratified CV on the training split...")
    t0 = time.time()
    cv_results = select_hyperparams_via_cv(model_key, X_train, y_train, log=log)
    log(f"  CV took {time.time() - t0:.1f}s")
    save_table(cv_results, f"{name}_{model_key}_cv_selection.csv")

    param_names = list(MODEL_REGISTRY[model_key]["param_grid"][0].keys())
    best_row = cv_results.loc[cv_results["f1_macro"].idxmax()]
    best_params = {p: (None if pd.isna(best_row[p]) else best_row[p]) for p in param_names}
    # numpy/pandas can upcast int params (e.g. n_estimators) to float when a
    # column also holds None; cast back so the estimator gets ints, not floats.
    best_params = {p: (int(v) if isinstance(v, (float, np.floating)) and v is not None and float(v).is_integer() else v)
                   for p, v in best_params.items()}
    log(f"  best params = {best_params} (by CV macro-F1 = {best_row['f1_macro']:.4f})")

    model = MODEL_REGISTRY[model_key]["factory"](**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_metrics = summary_metrics(y_test, y_pred)
    baseline_acc = majority_baseline_accuracy(y_train, y_test)
    test_metrics["majority_class_baseline_accuracy"] = baseline_acc
    test_metrics["best_params"] = best_params
    test_metrics["granularity"] = name
    test_metrics["model"] = model_key
    log(f"  held-out KDDTest+ : accuracy={test_metrics['accuracy']:.4f}  "
        f"f1_macro={test_metrics['f1_macro']:.4f}  f1_weighted={test_metrics['f1_weighted']:.4f}  "
        f"(majority-class baseline accuracy={baseline_acc:.4f})")

    labels = sorted(set(y_train) | set(y_test))
    report = per_class_report(y_test, y_pred, labels=labels)
    save_table(report, f"{name}_{model_key}_per_class_report_testset.csv")
    save_confusion_matrix_plot(
        y_test, y_pred, labels,
        title=f"{name} / {model_key} — confusion matrix on KDDTest+",
        filename=f"{name}_{model_key}_confusion_matrix_testset.png",
    )

    return {
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "per_class_report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_all(log=print, model_keys=("knn", "random_forest")) -> dict:
    log("Loading data...")
    train_df = load_train()
    test_df = load_test()
    log(f"  train: {train_df.shape}, test: {test_df.shape}")

    log("Fitting preprocessor on training split only...")
    pre = Preprocessor().fit(train_df)

    results = {}
    for name in ["binary", "category5", "multiclass"]:
        for model_key in model_keys:
            results[(name, model_key)] = run_granularity(name, model_key, train_df, test_df, pre, log=log)

    summary = pd.DataFrame([results[key]["test_metrics"] for key in results])
    cols = ["granularity", "model", "best_params", "accuracy", "majority_class_baseline_accuracy",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro"]
    summary = summary[cols]
    save_table(summary, "summary_all_granularities_testset.csv")
    log("\n=== Summary (held-out KDDTest+) ===")
    log(summary.to_string(index=False))

    return results


if __name__ == "__main__":
    run_all()
