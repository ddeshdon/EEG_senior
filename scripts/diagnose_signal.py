#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Univariate feature diagnostics (multi-class or binary).

- Filters rows by min duration and min good_window_frac
- If --positive-regex is given:
    -> collapses labels to 2 classes: POS vs NEG
       and runs Mann–Whitney U + Cliff's delta
    -> saves:  univariate_binary.csv
- Otherwise:
    -> keeps all group_label levels
       and runs Kruskal–Wallis across all classes
    -> saves:  univariate_multiclass.csv
- In both modes:
    -> makes boxplots for the top 8 features (smallest p-values)
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal


# Columns that are NOT features
META = {
    "file_id",
    "participant_id",
    "group_label",
    "group_label_raw",
    "duration_s",
    "good_window_frac",
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Small-sample Cliff's delta for effect size.
    Returns NaN if one of the groups is empty.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    more = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    return (more - less) / (nx * ny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="data/features/features_wide.csv",
                    help="Path to features_wide.csv")
    ap.add_argument("--min-duration", type=float, default=180.0,
                    help="Min duration_s to keep (default: 180)")
    ap.add_argument("--min-good", type=float, default=0.3,
                    help="Min good_window_frac to keep (default: 0.3)")
    ap.add_argument("--positive-regex", default=None,
                    help="Binary mode: regex for positive class in group_label (case-insensitive).")
    ap.add_argument("--outdir", default="data/features/diagnostics",
                    help="Output directory for CSV + boxplots.")
    args = ap.parse_args()

    wide_path = Path(args.wide)
    if not wide_path.exists():
        raise SystemExit(f"Wide file not found: {wide_path}")

    # ---------------- Load + basic quality filter ----------------
    df = pd.read_csv(wide_path)

    # Apply quality thresholds if columns exist
    if "duration_s" in df.columns:
        df = df[df["duration_s"].fillna(0) >= args.min_duration]
    if "good_window_frac" in df.columns:
        df = df[df["good_window_frac"].fillna(0) >= args.min_good]

    df = df[df["group_label"].notna()].reset_index(drop=True)
    print(f"[info] kept {len(df)} rows after quality filter")

    # ---------------- Label handling ----------------
    y = df["group_label"].astype(str)

    binary_mode = args.positive_regex is not None
    if binary_mode:
        pattern = args.positive_regex
        print(f"[info] Binary mode with POS regex: {pattern}")
        mask = y.str.contains(pattern, case=False, regex=True, na=False)
        # Map to POS / NEG
        y = mask.map({True: "POS", False: "NEG"})
        print("[info] binary class counts:")
        print(y.value_counts())

    # ---------------- Feature selection ----------------
    num_cols = [
        c for c in df.columns
        if c not in META and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"[info] numeric feature columns: {len(num_cols)}")

    # Ensure output dir exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---------------- Binary case (POS vs NEG) ----------------
    if binary_mode and y.nunique() == 2:
        classes = sorted(y.unique().tolist())  # e.g. ["NEG", "POS"]
        g1, g2 = classes[0], classes[1]
        print(f"[info] comparing {g1} vs {g2}")

        for c in num_cols:
            a = df.loc[y == g1, c].dropna().values
            b = df.loc[y == g2, c].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue  # skip features with too few samples

            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            d = cliffs_delta(a, b)

            rows.append((
                c,
                float(np.nanmedian(a)),
                float(np.nanmedian(b)),
                d,
                p,
                len(a),
                len(b),
            ))

        res = pd.DataFrame(
            rows,
            columns=[
                "feature",
                f"med_{g1}",
                f"med_{g2}",
                "cliffs_delta",
                "p_mannwhitney",
                f"n_{g1}",
                f"n_{g2}",
            ],
        )

        res = res.sort_values(["p_mannwhitney", "cliffs_delta"],
                              ascending=[True, False])
        out_csv = outdir / "univariate_binary.csv"
        res.to_csv(out_csv, index=False)
        print(f"→ wrote {out_csv}")

    # ---------------- Multi-class case (≥3 labels) ----------------
    else:
        labels = sorted(y.unique().tolist())
        print(f"[info] multi-class mode with groups: {labels}")

        for c in num_cols:
            groups = [df.loc[y == g, c].dropna().values for g in labels]
            # Need at least 3 samples in every group
            if any(len(g) < 3 for g in groups):
                continue

            pk = kruskal(*groups).pvalue
            rows.append((c, pk))

        res = pd.DataFrame(rows, columns=["feature", "p_kruskal"])
        res = res.sort_values("p_kruskal")
        out_csv = outdir / "univariate_multiclass.csv"
        res.to_csv(out_csv, index=False)
        print(f"→ wrote {out_csv}")

    # ---------------- Quick boxplots of top 8 features ----------------
    if not res.empty:
        top_feats = res.head(8)["feature"].tolist()
        labels_for_plot = list(sorted(y.unique().tolist()))

        for c in top_feats:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            data = [df.loc[y == g, c].dropna().values for g in labels_for_plot]
            ax.boxplot(data, labels=labels_for_plot)
            ax.set_title(c)
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(outdir / f"box_{c}.png", dpi=150)
            plt.close(fig)

        print(f"→ saved boxplots for top {len(top_feats)} features in {outdir}")

    else:
        print("[warn] no features passed the minimum sample requirement; no plots written.")


if __name__ == "__main__":
    main()
