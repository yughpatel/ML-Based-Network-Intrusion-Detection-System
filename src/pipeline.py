import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

# Cap on how far a rare class gets oversampled: min(count * SMOTE_GROWTH_CAP,
# SMOTE_TARGET_CEILING). A class this rare (52 U2R rows) can't responsibly be
# grown all the way up to "normal"'s size -- almost every neighbor SMOTE
# would interpolate between is itself synthetic by that point, so the model
# just overfits to noise instead of learning real structure.
SMOTE_GROWTH_CAP = 20
SMOTE_TARGET_CEILING = 2000


def apply_smote(X, y, log=print):
    """Oversample rare classes only (see cap above), one class at a time so
    each gets its own k_neighbors sized to its actual count -- a single
    imblearn SMOTE call applies one k_neighbors to every targeted class,
    which breaks classes as small as U2R's 52 rows. Classes with fewer than
    2 rows can't be SMOTE'd (there's nothing to interpolate between) and are
    left untouched."""
    X = np.asarray(X)
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    for cls, count in counts.items():
        target = min(count * SMOTE_GROWTH_CAP, SMOTE_TARGET_CEILING)
        if count >= target or count < 2:
            continue
        k_neighbors = min(5, count - 1)
        sm = SMOTE(sampling_strategy={cls: target}, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
        X, y = sm.fit_resample(X, y)
        log(f"    SMOTE: {cls} {count} -> {target} (k_neighbors={k_neighbors})")
    return X, y


# Each model is a (param -> estimator) factory plus the grid of param dicts to
# try via CV. Random forest uses class_weight="balanced" so rare classes
# (R2L, U2R) contribute as much to the split criterion as "normal"/"dos" do,
# instead of being swamped -- KNN has no such knob, which is why it misses
# almost all R2L/U2R on category5/multiclass (see per-class reports). The
# "_smote" variants reuse the same factory/grid but oversample rare classes
# in the training data first (see apply_smote / run_granularity).
_KNN = {
    "factory": lambda n_neighbors: KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
    "param_grid": [{"n_neighbors": k} for k in [1, 3, 5, 7, 11, 15, 21]],
}
_RANDOM_FOREST = {
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
}
_GRADIENT_BOOSTING = {
    # early_stopping="auto" (the default) carves out its own stratified
    # validation split internally, which blows up on classes with a
    # handful of rows total (e.g. multiclass's "spy" has 2) -- external CV
    # already guards against overfitting here, so early stopping is disabled.
    "factory": lambda max_iter, max_depth: HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        class_weight="balanced",
        early_stopping=False,
        random_state=RANDOM_STATE,
    ),
    "param_grid": [
        {"max_iter": 100, "max_depth": None},
        {"max_iter": 100, "max_depth": 10},
        {"max_iter": 200, "max_depth": None},
        {"max_iter": 300, "max_depth": None},
    ],
}
MODEL_REGISTRY = {
    "knn": {**_KNN, "smote": False},
    "knn_smote": {**_KNN, "smote": True},
    "random_forest": {**_RANDOM_FOREST, "smote": False},
    "random_forest_smote": {**_RANDOM_FOREST, "smote": True},
    "gradient_boosting": {**_GRADIENT_BOOSTING, "smote": False},
    "gradient_boosting_smote": {**_GRADIENT_BOOSTING, "smote": True},
}


def _cv_folds_for(y) -> int:
    min_class_count = pd.Series(y).value_counts().min()
    return max(2, min(MAX_CV_FOLDS, int(min_class_count)))


def select_hyperparams_via_cv(model_key: str, X, y, log=print) -> pd.DataFrame:
    factory = MODEL_REGISTRY[model_key]["factory"]
    param_grid = MODEL_REGISTRY[model_key]["param_grid"]
    use_smote = MODEL_REGISTRY[model_key]["smote"]
    n_splits = _cv_folds_for(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = np.asarray(X)
    y = np.asarray(y)

    rows = []
    for params in param_grid:
        fold_metrics = []
        for train_idx, val_idx in skf.split(X, y):
            # SMOTE is fit on the training fold only -- the validation fold
            # must stay untouched real data, or its score would be inflated
            # by synthetic points leaking distributional info from itself.
            X_fit, y_fit = (X[train_idx], y[train_idx])
            if use_smote:
                X_fit, y_fit = apply_smote(X_fit, y_fit, log=lambda *_: None)
            model = factory(**params)
            model.fit(X_fit, y_fit)
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

    X_fit, y_fit = (X_train, y_train)
    if MODEL_REGISTRY[model_key]["smote"]:
        log("  applying SMOTE to the full training split before the final fit...")
        X_fit, y_fit = apply_smote(np.asarray(X_train), np.asarray(y_train), log=log)

    model = MODEL_REGISTRY[model_key]["factory"](**best_params)
    model.fit(X_fit, y_fit)
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


def run_all(log=print, model_keys=("knn", "random_forest", "gradient_boosting",
                                    "knn_smote", "random_forest_smote", "gradient_boosting_smote")) -> dict:
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
