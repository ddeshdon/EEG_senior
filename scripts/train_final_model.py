#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train final binary model: Experienced vs Non-experienced meditator.

Pipeline:
  - Load features_wide.csv
  - Filter by duration_s and good_window_frac
  - Map 4 labels -> 2 labels:
        Non-experienced: beginner, never_practiced
        Experienced    : trained_over_3_years, meditation_master_or_instructor
  - Impute missing values (median) + RobustScaler
  - L1-Logistic Regression for feature selection
  - Train CatBoostClassifier with class_weights
  - 5-fold Stratified CV (report accuracy, balanced accuracy, F1, AUC)
  - Fit calibrated classifier (isotonic) on all data
  - Compute global SHAP importance for selected features
  - Save model bundle with preprocessing + classifier via joblib
  - Save SHAP importance as JSON

Usage:
  python scripts/train_final_model_catboost.py \
    --wide data/features/features_wide.csv \
    --min-duration 180 --min-good 0.3 \
    --models-dir models

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import shap

from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

META_COLS = {"file_id", "participant_id", "group_label", "duration_s", "good_window_frac"}


# -------------------- Helpers --------------------


def load_and_filter(wide_path: Path, min_duration: float, min_good: float) -> pd.DataFrame:
    """Load features_wide.csv and apply duration / quality filters."""
    df = pd.read_csv(wide_path)
    if "group_label" not in df.columns:
        raise SystemExit("features_wide.csv is missing 'group_label' column.")

    before = len(df)
    if "duration_s" in df.columns:
        df = df[df["duration_s"].fillna(0) >= float(min_duration)]
    if "good_window_frac" in df.columns:
        df = df[df["good_window_frac"].fillna(0) >= float(min_good)]
    df = df[df["group_label"].notna()].reset_index(drop=True)
    after = len(df)

    print(f"[info] quality filter kept {after}/{before} rows "
          f"(min_duration={min_duration}, min_good={min_good})")
    print("[info] labels after filtering:")
    print(df["group_label"].value_counts())

    if len(df) < 10:
        print("[warn] Very small dataset after filtering.")
    return df


def map_to_binary_labels(df: pd.DataFrame) -> pd.Series:
    """
    Map 4 meditation groups into 2:

        Non-experienced (0): beginner, never_practiced
        Experienced     (1): trained_over_3_years, meditation_master_or_instructor
    """
    raw = df["group_label"].astype(str).str.strip()

    non_exp = {"beginner", "never_practiced"}
    exp = {"trained_over_3_years", "meditation_master_or_instructor"}

    y = []
    unknown = set()
    for g in raw:
        if g in non_exp:
            y.append(0)
        elif g in exp:
            y.append(1)
        else:
            y.append(None)
            unknown.add(g)

    if unknown:
        raise SystemExit(f"[error] Unknown labels encountered (cannot map to binary): {unknown}")

    y = pd.Series(y, index=df.index, name="binary_label")
    print("[info] binary label counts (0=non_exp,1=exp):")
    print(y.value_counts())
    return y


def make_numeric_feature_matrix(df: pd.DataFrame, include_quality: bool = False):
    """Extract numeric feature columns into X (no impute/scale yet)."""
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if not include_quality and c in ("duration_s", "good_window_frac"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise SystemExit("No numeric feature columns found.")

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, cols


def l1_feature_selection(X_scaled: np.ndarray, y: pd.Series, feature_names, min_keep: int = 5):
    """
    Use L1-penalized logistic regression as a simple feature selector.

    - Fit L1-logreg with class_weight='balanced'
    - Keep features with non-zero mean |coef|
    - If that is < min_keep, keep top min_keep by coefficient magnitude
    """
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=500,
    )
    clf.fit(X_scaled, y)

    coef = np.abs(clf.coef_).mean(axis=0)
    nonzero_mask = coef > 1e-6
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero == 0:
        print("[warn] L1-logreg selected 0 features -> using all.")
        mask = np.ones_like(coef, dtype=bool)
    elif n_nonzero < min_keep:
        print(f"[info] L1-logreg non-zero features ({n_nonzero}) < min_keep ({min_keep}). "
              f"Keeping top {min_keep} by |coef|.")
        idx_sorted = np.argsort(coef)[::-1]
        mask = np.zeros_like(coef, dtype=bool)
        mask[idx_sorted[:min_keep]] = True
    else:
        mask = nonzero_mask

    selected_features = [f for f, keep in zip(feature_names, mask) if keep]
    print(f"[info] selected {len(selected_features)}/{len(feature_names)} features via L1-logreg.")
    return mask, selected_features, coef


def build_catboost(class_weights):
    """Create a CatBoostClassifier configured for this project."""
    return CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_state=42,
        verbose=False,
    )


# -------------------- Main training script --------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="data/features/features_wide.csv",
                    help="Path to features_wide.csv")
    ap.add_argument("--min-duration", type=float, default=180.0,
                    help="Min duration_s to keep")
    ap.add_argument("--min-good", type=float, default=0.3,
                    help="Min good_window_frac to keep")
    ap.add_argument("--include-quality-features", action="store_true",
                    help="Include duration_s & good_window_frac as features")
    ap.add_argument("--models-dir", default="models",
                    help="Directory to save model + SHAP importance")
    args = ap.parse_args()

    wide_path = Path(args.wide)
    if not wide_path.exists():
        raise SystemExit(f"wide file not found: {wide_path}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] models will be saved to: {models_dir}")

    # 1) Load & filter
    df = load_and_filter(wide_path, args.min_duration, args.min_good)
    if len(df) < 10:
        raise SystemExit("[error] Too few rows for a meaningful model. Collect more data or relax filters.")

    # 2) Map labels -> binary (0/1)
    y = map_to_binary_labels(df)
    if y.nunique() != 2:
        raise SystemExit("[error] Need exactly 2 classes for this script.")

    # 3) Numeric features
    X_raw, feature_names = make_numeric_feature_matrix(df, include_quality=args.include_quality_features)

    # 4) Impute + scale
    imp = SimpleImputer(strategy="median")
    scl = RobustScaler()

    X_imp = imp.fit_transform(X_raw)
    X_scaled = scl.fit_transform(X_imp)

    # 5) L1 feature selection
    mask, selected_features, l1_coef = l1_feature_selection(X_scaled, y, feature_names)
    X_sel = X_scaled[:, mask]

    # 6) Class weights for CatBoost (to handle imbalance)
    counts = y.value_counts().sort_index()  # index 0,1
    n_total = float(len(y))
    w0 = n_total / (2.0 * counts.loc[0])
    w1 = n_total / (2.0 * counts.loc[1])
    class_weights = [w0, w1]
    print(f"[info] class weights for CatBoost: {class_weights}")

    # 7) 5-fold Stratified CV for performance estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, baccs, f1s, aucs = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_sel, y), start=1):
        model_cv = build_catboost(class_weights)
        model_cv.fit(X_sel[tr], y.iloc[tr])

        y_pred = model_cv.predict(X_sel[te])
        y_proba = model_cv.predict_proba(X_sel[te])[:, 1]

        accs.append(accuracy_score(y.iloc[te], y_pred))
        baccs.append(balanced_accuracy_score(y.iloc[te], y_pred))
        f1s.append(f1_score(y.iloc[te], y_pred))
        try:
            aucs.append(roc_auc_score(y.iloc[te], y_proba))
        except ValueError:
            aucs.append(np.nan)

        print(f"[cv] fold {fold}: "
              f"acc={accs[-1]:.3f}, bacc={baccs[-1]:.3f}, "
              f"f1={f1s[-1]:.3f}, auc={aucs[-1]:.3f}")

    cv_metrics = {
        "accuracy_mean": float(np.nanmean(accs)),
        "accuracy_std": float(np.nanstd(accs)),
        "balanced_accuracy_mean": float(np.nanmean(baccs)),
        "balanced_accuracy_std": float(np.nanstd(baccs)),
        "f1_mean": float(np.nanmean(f1s)),
        "f1_std": float(np.nanstd(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
    }

    print("\n[cv] summary (5-fold, Experienced vs Non-experienced):")
    for k, v in cv_metrics.items():
        print(f"  {k}: {v:.3f}")

    # 8) Fit final base CatBoost on ALL data (for SHAP and as base for calibration)
    base_model = build_catboost(class_weights)
    base_model.fit(X_sel, y)

    # 9) Calibrated classifier (isotonic)
    #    This is the one you will normally use for probabilities in deployment.
    calib_base = build_catboost(class_weights)
    calib = CalibratedClassifierCV(
        base_estimator=calib_base,
        method="isotonic",
        cv=5,
    )
    calib.fit(X_sel, y)

    # 10) SHAP global importance (using the base_model)
    print("\n[info] computing SHAP global feature importance (this may take a bit)...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sel)

    # For binary classification, shap_values can be:
    # - array (n_samples, n_features), or
    # - list of arrays [class0, class1]. We'll use class1 if list.
    if isinstance(shap_values, list):
        # pick class 1 (experienced)
        shap_vals_use = shap_values[1]
    else:
        shap_vals_use = shap_values

    mean_abs_shap = np.mean(np.abs(shap_vals_use), axis=0)
    shap_importance = {
        feat: float(val)
        for feat, val in zip(selected_features, mean_abs_shap)
    }

    # Sort for readability
    shap_importance_sorted = dict(
        sorted(shap_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    # 11) Build model bundle and save
    model_bundle = {
        "imputer": imp,
        "scaler": scl,
        "feature_names": feature_names,
        "feature_selector_mask": mask,
        "selected_features": selected_features,
        "l1_coef": l1_coef.tolist(),
        "classifier_calibrated": calib,
        "class_weights": class_weights,
        "class_counts": counts.to_dict(),
        "label_mapping": {0: "non_experienced", 1: "experienced"},
        "cv_metrics": cv_metrics,
        "shap_global_importance": shap_importance_sorted,
    }

    model_path = models_dir / "experienced_catboost_model.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Saved model bundle to {model_path}")

    # 12) Save SHAP importance + CV metrics as JSON for dashboard
    json_path = models_dir / "experienced_shap_importance.json"
    json_path.write_text(json.dumps(shap_importance_sorted, indent=2, ensure_ascii=False))
    print(f"✅ Saved SHAP global importance to {json_path}")

    metrics_path = models_dir / "experienced_cv_metrics.json"
    metrics_path.write_text(json.dumps(cv_metrics, indent=2, ensure_ascii=False))
    print(f"✅ Saved CV metrics to {metrics_path}")


if __name__ == "__main__":
    main()
