#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train and compare SVM(linear), RandomForest, Logistic Regression on features_wide.csv
with robust handling for Thai labels, imbalanced classes, and plotting.

Usage examples:
  # Multi-class with plots
  python scripts/classify.py \
    --wide data/features/features_wide.csv \
    --min-duration 180 --min-good 0.3 --plots

  # Binary: mark 'ฝึกมากกว่า 3 ปี' OR 'เกจิ หรือ อจ.สอนสมาธิ' as positive (regex, case-insensitive)
  python scripts/classify.py \
    --positive-regex "(ฝึกมากกว่า 3 ปี|เกจิ|อจ\.สอนสมาธิ)" \
    --plots --max-boxplots 12

  # Binary: substring (case-insensitive) instead of regex
  python scripts/classify.py \
    --binary-substring "เริ่มต้น" --plots
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

META_COLS = {"file_id", "participant_id", "group_label"}


# ---------------- IO & filtering ----------------

def load_and_filter(wide_path: Path, min_duration: float, min_good: float) -> pd.DataFrame:
    """Load features_wide.csv and apply basic quality filters."""
    df = pd.read_csv(wide_path)
    if "group_label" not in df.columns:
        raise SystemExit("features_wide.csv is missing 'group_label' column.")

    before = len(df)
    if "duration_s" in df.columns:
        df = df[df["duration_s"].fillna(0) >= float(min_duration)]
    if "good_window_frac" in df.columns:
        df = df[df["good_window_frac"].fillna(0) >= float(min_good)]
    after = len(df)

    print(f"[info] quality filter kept {after}/{before} rows "
          f"(min_duration={min_duration}, min_good={min_good})")

    df = df[df["group_label"].notna()].reset_index(drop=True)

    print("[info] labels after filtering:")
    print(df["group_label"].value_counts())

    if len(df) < 3:
        print("[warn] Very small dataset after filtering.")
    return df


def make_features(df: pd.DataFrame, include_quality_as_features: bool = False):
    """Select numeric feature columns (optionally excluding duration_s / good_window_frac)."""
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if not include_quality_as_features and c in ("duration_s", "good_window_frac"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise SystemExit("No numeric feature columns found.")

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, cols


def choose_cv(y: pd.Series):
    """
    Choose a cross-validation scheme that is safe for imbalanced, small-N data.

    - If the smallest class has < 5 samples → use Leave-One-Out CV.
    - Otherwise, use StratifiedKFold with n_splits ≤ min(class counts, 5).
    """
    n = len(y)
    cls_counts = y.value_counts()
    min_per_class = int(cls_counts.min())
    n_classes = len(cls_counts)

    print(f"[info] total samples: {n}, n_classes: {n_classes}, class counts: {cls_counts.to_dict()}")

    # Very small / imbalanced case: safest is LOOCV
    if min_per_class < 5:
        print(f"[info] Using Leave-One-Out CV (min class size = {min_per_class})")
        return LeaveOneOut()

    # Otherwise: stratified K-fold, but never more folds than the smallest class
    n_splits = min(5, min_per_class)
    n_splits = max(2, n_splits)
    print(f"[info] Using StratifiedKFold with {n_splits} splits")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ---------------- Plot helpers ----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, classes, title, outpath: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_metrics_bar(results, outpath: Path):
    labels = [r["model"] for r in results]
    acc = [r["accuracy"] for r in results]
    bacc = [r["balanced_accuracy"] for r in results]
    f1m = [r["macro_f1"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x - width, acc, width, label='Accuracy')
    ax.bar(x, bacc, width, label='Balanced Acc')
    ax.bar(x + width, f1m, width, label='Macro F1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_top_features_bar(values, feat_names, title, outpath: Path, topk=10):
    idx = np.argsort(np.abs(values))[::-1][:topk]
    fig = plt.figure(figsize=(8, max(3, 0.35 * topk + 1)))
    ax = fig.add_subplot(111)
    y = np.arange(len(idx))
    ax.barh(y, np.abs(values[idx]))
    ax.set_yticks(y)
    ax.set_yticklabels([feat_names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_roc_binary(y_true_bin, y_score, title, outpath: Path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_feature_boxplots(df, features, label_col, outdir: Path, max_plots=12):
    ensure_dir(outdir)
    n = min(len(features), max_plots)
    for f in features[:n]:
        if f not in df.columns:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        groups = []
        labels = []
        for g in df[label_col].unique():
            labels.append(str(g))
            groups.append(df.loc[df[label_col] == g, f].dropna().values)
        ax.boxplot(groups, labels=labels)
        ax.set_title(f"{f} by {label_col}")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(outdir / f"box_{f}.png", dpi=150)
        plt.close(fig)


# ---------------- Evaluation ----------------

def evaluate_model(name, model, X, y, cv, plots_dir: Path = None):
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))

    print("\n==============================")
    print(f"{name} results")
    print("------------------------------")
    print(f"Accuracy          : {acc:.3f}")
    print(f"Balanced Accuracy : {bacc:.3f}")
    print(f"Macro F1          : {f1m:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))
    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    if plots_dir is not None:
        plot_confusion_matrix(cm, list(np.unique(y)),
                              f"Confusion: {name}",
                              plots_dir / f"cm_{name}.png")

    return {
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": f1m
    }, y_pred


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="data/features/features_wide.csv",
                    help="Path to features_wide.csv")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="Min duration_s to keep (e.g., 180)")
    ap.add_argument("--min-good", type=float, default=0.0,
                    help="Min good_window_frac to keep (e.g., 0.3)")
    ap.add_argument("--include-quality-features", action="store_true",
                    help="Include duration_s and good_window_frac as model features")
    ap.add_argument("--binary-substring", default=None,
                    help="Collapse labels to binary by literal substring in group_label (case-insensitive)")
    ap.add_argument("--positive-regex", default=None,
                    help=("Regex for POSITIVE class in group_label (case-insensitive). "
                          "Example: '(ฝึกมากกว่า 3 ปี|เกจิ|อจ\\.สอนสมาธิ)'"))
    ap.add_argument("--plots", action="store_true", help="Save plots")
    ap.add_argument("--plots-dir", default=None,
                    help="Directory to save plots (default next to wide)")
    ap.add_argument("--max-boxplots", type=int, default=0,
                    help="Save up to N per-feature boxplots by label (0 disables)")
    ap.add_argument("--out", default=None,
                    help="Where to save model_results.csv (default next to wide)")
    args = ap.parse_args()

    wide_path = Path(args.wide)
    if not wide_path.exists():
        raise SystemExit(f"Wide file not found: {wide_path}")

    plots_dir = None
    if args.plots:
        plots_dir = Path(args.plots_dir) if args.plots_dir else (wide_path.parent / "plots")
        ensure_dir(plots_dir)
        print(f"[info] plots will be saved to: {plots_dir}")

    # Load + filter
    df = load_and_filter(wide_path, args.min_duration, args.min_good)

    # Labels (binary or multi)
    y_raw = df["group_label"].astype(str)
    is_binary = False
    pos_name = neg_name = None

    if args.positive_regex or args.binary_substring:
        if args.positive_regex:
            pattern = args.positive_regex
        else:
            pattern = re.escape(args.binary_substring)
        mask = y_raw.str.contains(pattern, case=False, na=False, regex=True)
        pos_name = f"Has/{pattern}"
        neg_name = f"No/{pattern}"
        y = mask.map({True: pos_name, False: neg_name})
        is_binary = True
        print(f"[info] Binary mode: positives = group_label matching regex: {pattern}")
    else:
        y = y_raw.copy()

    # Guards: need at least 2 classes, and ≥2 samples per class for CV
    cls_counts = y.value_counts()
    if len(cls_counts) < 2:
        raise SystemExit(
            f"[error] Only one class present after filters: {cls_counts.to_dict()}. "
            f"Relax --min-duration/--min-good, change regex/substring, or run multi-class."
        )
    if cls_counts.min() < 2:
        raise SystemExit(
            f"[error] Each class needs >= 2 samples for CV. "
            f"Counts: {cls_counts.to_dict()}. "
            f"Relax filters or collect more data."
        )

    # Features
    X, feat_names = make_features(df, include_quality_as_features=args.include_quality_features)

    # CV scheme
    cv = choose_cv(y)

    # Models (all class-weighted to handle imbalance)
    svm_lin = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        SVC(kernel="linear", class_weight="balanced",
            probability=False, random_state=42),
    )
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample"
        ),
    )
    logreg = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=200
        ),
    )

    results = []
    preds = {}

    for name, model in [
        ("SVM(linear)", svm_lin),
        ("RandomForest", rf),
        ("LogisticRegression", logreg),
    ]:
        res, y_pred = evaluate_model(name, model, X, y, cv, plots_dir=plots_dir)
        results.append(res)
        preds[name] = y_pred

    # Fit on full data (ONLY for feature introspection plots)
    svm_lin.fit(X, y)
    logreg.fit(X, y)
    rf.fit(X, y)

    # Top features / importances
    if args.plots:
        if hasattr(svm_lin[-1], "coef_"):
            w = np.mean(np.abs(svm_lin[-1].coef_), axis=0)
            plot_top_features_bar(
                w, feat_names,
                "Top |weights|: SVM(linear)",
                plots_dir / "top_features_SVM(linear).png",
            )
        if hasattr(logreg[-1], "coef_"):
            w = np.mean(np.abs(logreg[-1].coef_), axis=0)
            plot_top_features_bar(
                w, feat_names,
                "Top |weights|: LogisticRegression",
                plots_dir / "top_features_LogisticRegression.png",
            )
        if hasattr(rf[-1], "feature_importances_"):
            imp = rf[-1].feature_importances_
            plot_top_features_bar(
                imp, feat_names,
                "Top importances: RandomForest",
                plots_dir / "top_features_RandomForest.png",
            )

        # Metrics comparison
        plot_metrics_bar(results, plots_dir / "model_comparison.png")

        # ROC (binary only)
        if is_binary and pos_name is not None:
            # Build binary ground truth (1 = positive class)
            y_bin = (y == pos_name).astype(int).values

            # SVM decision function
            if hasattr(svm_lin[-1], "decision_function"):
                y_score = cross_val_predict(svm_lin, X, y, cv=cv, method='decision_function')
                plot_roc_binary(y_bin, y_score, "ROC: SVM(linear)",
                                plots_dir / "roc_SVM(linear).png")

            # Logistic regression decision function
            if hasattr(logreg[-1], "decision_function"):
                y_score = cross_val_predict(logreg, X, y, cv=cv, method='decision_function')
                plot_roc_binary(y_bin, y_score, "ROC: LogisticRegression",
                                plots_dir / "roc_LogisticRegression.png")

            # RandomForest probability for positive class
            if hasattr(rf[-1], "predict_proba"):
                prob = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')
                classes = rf[-1].classes_
                if len(classes) == 2:
                    pos_idx = list(classes).index(pos_name)
                    y_score = prob[:, pos_idx]
                    plot_roc_binary(y_bin, y_score, "ROC: RandomForest",
                                    plots_dir / "roc_RandomForest.png")

        # Optional per-feature boxplots
        if args.max_boxplots > 0:
            plot_feature_boxplots(
                df.assign(_label=y),
                feat_names,
                "_label",
                plots_dir / "boxplots",
                max_plots=args.max_boxplots,
            )

    # Save metrics table
    out_path = Path(args.out) if args.out else (wide_path.parent / "model_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n✅ Saved metrics to {out_path}")
    if args.plots:
        print(f"✅ Plots saved in {plots_dir}")


if __name__ == "__main__":
    main()
