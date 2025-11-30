#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make easy-to-read 2D cluster plots from features_wide.csv:
- PCA (2D)
- t-SNE (2D)
- (optional) UMAP (2D, if installed)

Each point = participant, color = group_label (or binary mapping).
Adds class centroids and convex hulls for readability.

Usage examples
--------------

# Multi-class, filter short/noisy, save plots
python scripts/visualize_clusters.py \
  --wide data/features/features_wide.csv \
  --min-duration 180 --min-good 0.3 \
  --outdir data/features/plots

# Binary: Experienced vs Non-experienced (English labels)
# Experienced = trained_over_3_years OR meditation_master_or_instructor
python scripts/visualize_clusters.py \
  --wide data/features/features_wide.csv \
  --min-duration 180 --min-good 0.3 \
  --positive-regex "(trained_over_3_years|meditation_master_or_instructor)" \
  --outdir data/features/plots

# Binary: substring (case-insensitive) instead of regex
python scripts/visualize_clusters.py \
  --wide data/features/features_wide.csv \
  --min-duration 180 --min-good 0.3 \
  --binary-substring "beginner" \
  --outdir data/features/plots
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional UMAP
try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional convex hull
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

META_COLS = {"file_id", "participant_id", "group_label"}


# ---------------- IO & filtering ----------------

def load_and_filter(wide_path: Path,
                    min_duration: float,
                    min_good: float) -> pd.DataFrame:
    """Load features_wide.csv and apply basic quality filters."""
    df = pd.read_csv(wide_path)
    if "group_label" not in df.columns:
        raise SystemExit("wide file is missing 'group_label' column.")

    before = len(df)
    if "duration_s" in df.columns:
        df = df[df["duration_s"].fillna(0) >= float(min_duration)]
    if "good_window_frac" in df.columns:
        df = df[df["good_window_frac"].fillna(0) >= float(min_good)]
    df = df[df["group_label"].notna()].reset_index(drop=True)
    after = len(df)

    print(f"[info] kept {after}/{before} rows after quality filters "
          f"(min_duration={min_duration}, min_good={min_good})")

    print("[info] labels after filtering:")
    print(df["group_label"].value_counts())
    return df


def make_features(df: pd.DataFrame,
                  include_quality: bool = False):
    """
    Select numeric feature columns (optionally excluding duration_s / good_window_frac),
    impute missing values with median, and robust-scale the features.
    """
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if not include_quality and c in ("duration_s", "good_window_frac"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if not cols:
        raise SystemExit("No numeric features found in wide file.")

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scl = RobustScaler()
    X_scaled = scl.fit_transform(X_imp)

    X_out = pd.DataFrame(X_scaled, columns=cols, index=df.index)
    return X_out, cols


def map_labels(df: pd.DataFrame,
               positive_regex: str = None,
               binary_substring: str = None):
    """
    Map group_label to either:
      - Original multi-class labels (no args), or
      - Binary labels using regex / substring.
    """
    y_raw = df["group_label"].astype(str)

    if positive_regex or binary_substring:
        pattern = positive_regex if positive_regex else re.escape(binary_substring)
        mask = y_raw.str.contains(pattern,
                                  case=False,
                                  na=False,
                                  regex=True)
        pos_name = "POS"
        neg_name = "NEG"
        y = mask.map({True: pos_name, False: neg_name}).astype(str)
        print(f"[info] Binary mode enabled. POS if group_label matches: {pattern}")
        print("[info] Binary class counts:")
        print(y.value_counts())
        return y, True

    # Multi-class case
    y = y_raw.copy()
    print("[info] Using original multi-class group_label values.")
    print("[info] Class counts:")
    print(y.value_counts())
    return y, False


# ---------------- Plot helpers ----------------

def _convex_hull(ax, pts: np.ndarray, edge_alpha: float = 0.25):
    """Draw convex hull around a cloud of points (optional)."""
    if not HAS_SCIPY:
        return
    if pts.shape[0] < 3:
        return
    try:
        hull = ConvexHull(pts)
        cyc = np.append(hull.vertices, hull.vertices[0])
        ax.plot(pts[cyc, 0], pts[cyc, 1], linewidth=1, alpha=edge_alpha)
    except Exception:
        # Be robust to numerical issues
        pass


def _scatter(ax, Z: np.ndarray, labels: pd.Series, title: str):
    """
    Generic scatter plotting:
      - each class with its own color
      - class centroid (X marker)
      - convex hull (if SciPy available)
    """
    classes = pd.Index(sorted(labels.unique()))

    for lab in classes:
        idx = (labels == lab).values
        pts = Z[idx, :]

        # scatter points
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=40,
                   label=str(lab),
                   alpha=0.8)

        # centroid
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        ax.scatter([cx], [cy], s=120, marker="X")

        # convex hull outline
        _convex_hull(ax, pts)

    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


# ---------------- Embedding methods ----------------

def do_pca(X: np.ndarray, labels: pd.Series, outpng: Path):
    """2D PCA embedding."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var_exp = float(pca.explained_variance_ratio_.sum())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _scatter(ax, Z, labels, f"PCA (2D) — var exp: {var_exp:.2f}")
    fig.tight_layout()
    fig.savefig(outpng, dpi=160)
    plt.close(fig)
    print(f"[ok] saved {outpng}")


def do_tsne(X: np.ndarray, labels: pd.Series, outpng: Path):
    """2D t-SNE embedding (for small datasets)."""
    n = X.shape[0]
    if n < 5:
        print("[skip] t-SNE: need >=5 points to look meaningful.")
        return

    # choose perplexity sensibly for small-N
    perpl = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perpl,
        random_state=42,
        init="pca",
        n_iter=1000,
    )
    Z = tsne.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _scatter(ax, Z, labels, f"t-SNE (2D) — perplexity={perpl}")
    fig.tight_layout()
    fig.savefig(outpng, dpi=160)
    plt.close(fig)
    print(f"[ok] saved {outpng}")


def do_umap(X: np.ndarray, labels: pd.Series, outpng: Path):
    """2D UMAP embedding (optional)."""
    if not HAS_UMAP:
        print("[skip] UMAP not installed. `pip install umap-learn` to enable.")
        return

    n = X.shape[0]
    n_neighbors = min(15, max(5, n // 2))

    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=0.1,
    )
    Z = reducer.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _scatter(ax, Z, labels, f"UMAP (2D) — n_neighbors={n_neighbors}")
    fig.tight_layout()
    fig.savefig(outpng, dpi=160)
    plt.close(fig)
    print(f"[ok] saved {outpng}")


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
                    help="Include duration_s & good_window_frac as model features")
    ap.add_argument("--positive-regex", default=None,
                    help="Regex for POSITIVE class in group_label (case-insensitive).")
    ap.add_argument("--binary-substring", default=None,
                    help="Literal substring for POSITIVE class (case-insensitive).")
    ap.add_argument("--outdir", default=None,
                    help="Directory to save plots (default next to wide)")
    args = ap.parse_args()

    wide_path = Path(args.wide)
    if not wide_path.exists():
        raise SystemExit(f"wide file not found: {wide_path}")

    outdir = Path(args.outdir) if args.outdir else (wide_path.parent / "plots")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] saving plots to: {outdir}")

    # 1) Load + quality filter
    df = load_and_filter(wide_path,
                         min_duration=args.min_duration,
                         min_good=args.min_good)
    if len(df) < 3:
        raise SystemExit("[error] too few rows after filtering for a meaningful plot.")

    # 2) Map labels (multi-class or binary)
    y, is_binary = map_labels(df,
                              positive_regex=args.positive_regex,
                              binary_substring=args.binary_substring)

    # 3) Features
    X, feat_names = make_features(df,
                                  include_quality=args.include_quality_features)
    X_np = X.values

    # 4) PCA
    do_pca(X_np, y, outdir / "embed_pca.png")

    # 5) t-SNE
    do_tsne(X_np, y, outdir / "embed_tsne.png")

    # 6) UMAP (optional)
    do_umap(X_np, y, outdir / "embed_umap.png")


if __name__ == "__main__":
    main()
