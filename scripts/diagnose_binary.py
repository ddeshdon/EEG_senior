#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu  # non-parametric 2-group test
import matplotlib.pyplot as plt

META_COLS = {
    "file_id",
    "participant_id",
    "group_label",
    "group_label_raw",
    "duration_s",
    "good_window_frac",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main(args):
    wide_path = Path(args.wide)
    if not wide_path.exists():
        raise SystemExit(f"Wide file not found: {wide_path}")

    df = pd.read_csv(wide_path)

    # ---- merge to 2 groups: experienced vs non_experienced ----
    experienced_labels = [
        "trained_over_3_years",
        "meditation_master_or_instructor",
    ]
    df["group_bin"] = np.where(
        df["group_label"].isin(experienced_labels),
        "experienced",
        "non_experienced",
    )

    print("[info] counts in binary groups:")
    print(df["group_bin"].value_counts())

    # keep only rows with a valid group
    df = df[df["group_bin"].isin(["experienced", "non_experienced"])].copy()

    # choose numeric features (you can tweak META_COLS)
    feat_cols = [
        c for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[info] testing {len(feat_cols)} features")

    results = []
    for feat in feat_cols:
        g1 = df.loc[df["group_bin"] == "experienced", feat].dropna()
        g0 = df.loc[df["group_bin"] == "non_experienced", feat].dropna()

        if len(g1) < 3 or len(g0) < 3:
            continue  # too few samples

        # Mann–Whitney U test (non-parametric, 2 groups)
        stat, p = mannwhitneyu(g1, g0, alternative="two-sided")

        med1 = float(np.median(g1))
        med0 = float(np.median(g0))
        diff = med1 - med0  # simple effect size (you can add more later)

        results.append({
            "feature": feat,
            "median_experienced": med1,
            "median_non_experienced": med0,
            "median_diff_exp_minus_non": diff,
            "p_mannwhitney": p,
            "n_experienced": len(g1),
            "n_non_experienced": len(g0),
        })

    out_df = pd.DataFrame(results).sort_values("p_mannwhitney")
    out_path = wide_path.parent / "univariate_binary_mannwhitney.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ saved stats to {out_path}")

    # --- optional: boxplots for top-K features ---
    if args.max_boxplots > 0:
        plots_dir = wide_path.parent / "univariate_binary_plots"
        ensure_dir(plots_dir)

        top = out_df.head(args.max_boxplots)
        for _, row in top.iterrows():
            feat = row["feature"]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            data = [
                df.loc[df["group_bin"] == "experienced", feat].dropna(),
                df.loc[df["group_bin"] == "non_experienced", feat].dropna(),
            ]
            ax.boxplot(data, labels=["experienced", "non_experienced"])
            ax.set_title(feat)
            ax.set_xticklabels(["experienced", "non_experienced"], rotation=15)
            fig.tight_layout()
            fig.savefig(plots_dir / f"box_{feat}.png", dpi=150)
            plt.close(fig)

        print(f"📊 boxplots saved to {plots_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="data/features/features_wide.csv")
    ap.add_argument("--max-boxplots", type=int, default=10)
    args = ap.parse_args()
    main(args)
