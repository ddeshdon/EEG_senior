#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duration adequacy test for EEG meditation data.

- Re-extracts per-window features from raw/band CSVs (AF7/AF8/TP9/TP10).
- Checks feature stability vs cumulative duration (30..300 s).
- Computes split-half reliability.
- Trains simple classifiers at each duration to see accuracy vs duration.

Usage:
  # Multi-class (your English labels)
  python scripts/duration_adequacy.py \
    --manifest data/manifest.csv \
    --min-good 0.3 --min-duration 60 \
    --dur-grid 30,60,120,180,240,300

  # Binary: experienced-ish vs others
  python scripts/duration_adequacy.py \
    --manifest data/manifest.csv \
    --min-good 0.3 --min-duration 60 \
    --dur-grid 30,60,120,180,240,300 \
    --positive-regex "(trained_over_3_years|meditation_master_or_instructor)"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ------- config (keep in sync with your preprocess) -------
CHANS = ["TP9", "AF7", "AF8", "TP10"]
ALIASES = {
    "TP9":  ["RAW_TP9", "TP9", "EEG.TP9", "tp9", "TP9 "],
    "AF7":  ["RAW_AF7", "AF7", "EEG.AF7", "af7", "AF7 "],
    "AF8":  ["RAW_AF8", "AF8", "EEG.AF8", "af8", "AF8 "],
    "TP10": ["RAW_TP10", "TP10", "EEG.TP10", "tp10", "TP10 "],
}
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BANDPASS = (1, 45)
NOTCH_HZ = 50
WIN_S, WIN_OVERLAP = 4.0, 0.5  # 4s windows, 50% overlap

META_KEEP = ["file_id", "participant_id", "group_label", "csv_path", "sr_expected"]

# ------- helpers -------

def load_manifest(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    need = {"file_id", "participant_id", "group_label", "csv_path", "sr_expected", "channels_expected"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Manifest missing: {sorted(miss)}")
    return df


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def detect_schema(cols) -> str:
    cl = [c.lower() for c in cols]
    has_raw = all(name in cl for name in ["raw_tp9", "raw_af7", "raw_af8", "raw_tp10"])
    has_band = any(
        re.match(r"(delta|theta|alpha|beta|gamma)_(tp9|af7|af8|tp10)$", c, re.I)
        for c in cols
    )
    if has_raw:
        return "raw"
    if has_band:
        return "band"
    return "unknown"


def map_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[c for c in df.columns if c.strip().lower() in ("timestamp", "elements", "battery")],
        errors="ignore",
    )
    colmap = {}
    for canon, cands in ALIASES.items():
        found = next((c for c in cands if c in df.columns), None)
        if not found:
            raise ValueError(f"Missing raw channel '{canon}'")
        colmap[canon] = found

    out = df[[colmap[ch] for ch in CHANS]].copy()
    out.columns = CHANS
    for c in CHANS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any").reset_index(drop=True)
    return out


def butter_bandpass(low, high, fs, order=4):
    return butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")


def apply_filters(x: np.ndarray, fs: int) -> np.ndarray:
    # bandpass
    b, a = butter_bandpass(*BANDPASS, fs, order=4)
    xf = filtfilt(b, a, x, axis=0)

    # notch
    b, a = iirnotch(NOTCH_HZ, 30, fs)
    xf = filtfilt(b, a, xf, axis=0)

    return xf


def window_indices(n: int, fs: int, win_s: float = WIN_S, overlap: float = WIN_OVERLAP):
    w = int(win_s * fs)
    step = max(1, int(w * (1 - overlap)))
    idx = [(s, s + w) for s in range(0, n - w + 1, step)]
    return idx, w, step


def bandpower_welch(sig: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    f, Pxx = welch(sig, fs=fs, window="hamming", nperseg=int(fs),
                   noverlap=int(fs * 0.5))
    idx = (f >= fmin) & (f < fmax)
    return float(np.trapz(Pxx[idx], f[idx]))


def hjorth(sig: np.ndarray):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = np.var(sig)
    v1 = np.var(d1)
    v2 = np.var(d2)

    act = v0
    mob = np.sqrt(v1 / v0) if v0 > 0 else 0.0
    comp = np.sqrt(v2 / v1) / mob if (v1 > 0 and mob > 0) else 0.0
    return float(act), float(mob), float(comp)


def per_window_features(Xw: np.ndarray, fs: int) -> dict:
    # Xw: T x 4 window
    feats = {}
    for ci, ch in enumerate(CHANS):
        tot = bandpower_welch(Xw[:, ci], fs, 1, 45)
        bp = {n: bandpower_welch(Xw[:, ci], fs, *rng) for n, rng in BANDS.items()}

        # relative bandpowers
        for n, v in bp.items():
            feats[f"{n}_rel_{ch}"] = v / (tot + 1e-12)

        # ratios
        feats[f"theta_beta_ratio_{ch}"] = bp["theta"] / (bp["beta"] + 1e-12)
        feats[f"theta_alpha_ratio_{ch}"] = bp["theta"] / (bp["alpha"] + 1e-12)

        # Hjorth parameters
        act, mob, comp = hjorth(Xw[:, ci])
        feats[f"hjorth_activity_{ch}"] = act
        feats[f"hjorth_mobility_{ch}"] = mob
        feats[f"hjorth_complexity_{ch}"] = comp

    # frontal alpha asymmetry (AF7 vs AF8)
    feats["frontal_alpha_asym"] = (
        np.log(1e-12 + bandpower_welch(Xw[:, 1], fs, *BANDS["alpha"]))
        - np.log(1e-12 + bandpower_welch(Xw[:, 2], fs, *BANDS["alpha"]))
    )
    return feats


def extract_windows(df: pd.DataFrame, fs: int, schema: str) -> pd.DataFrame:
    """
    Returns a DataFrame of clean per-window features.
    - raw schema: re-window and compute all features
    - band schema: treat each row as a window and derive relative bands & ratios
    """
    if schema == "raw":
        X = map_raw(df).to_numpy(float)
        X = X - np.nanmean(X, axis=0)
        X = apply_filters(X, fs)

        idx, w, step = window_indices(len(X), fs, WIN_S, WIN_OVERLAP)
        rows = []
        for s, e in idx:
            W = X[s:e, :]

            # artifact rejection
            amp_bad = (np.max(np.abs(W), axis=0) > 150).any()

            zW = (W - W.mean(0)) / (W.std(0) + 1e-8)
            z_bad = (np.abs(zW) > 5).any()

            rel_high = []
            for ci in range(W.shape[1]):
                tot = bandpower_welch(W[:, ci], fs, 1, 45)
                hi = bandpower_welch(W[:, ci], fs, 35, 45)
                rel_high.append(hi / (tot + 1e-12))
            high_bad = (np.array(rel_high) > 0.5).any()

            if amp_bad or z_bad or high_bad:
                rows.append(None)
                continue

            rows.append(per_window_features(W, fs))

        Wdf = pd.DataFrame([r for r in rows if r is not None])
        return Wdf

    elif schema == "band":
        # If you only have band columns, approximate "windows" by rows.
        ok = [
            c for c in df.columns
            if re.match(r"^(Delta|Theta|Alpha|Beta|Gamma)_(TP9|AF7|AF8|TP10)$", c, re.I)
        ]
        if not ok:
            return pd.DataFrame()

        Z = df[ok].copy()
        for c in ok:
            Z[c] = pd.to_numeric(Z[c], errors="coerce")
        Z = Z.dropna(how="any")

        rows = []
        for _, r in Z.iterrows():
            feats = {}
            for ch in CHANS:
                tot = (
                    r[f"Delta_{ch}"]
                    + r[f"Theta_{ch}"]
                    + r[f"Alpha_{ch}"]
                    + r[f"Beta_{ch}"]
                    + r[f"Gamma_{ch}"]
                    + 1e-12
                )
                for b in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                    feats[f"{b.lower()}_rel_{ch}"] = r[f"{b}_{ch}"] / tot

                feats[f"theta_beta_ratio_{ch}"] = r[f"Theta_{ch}"] / (r[f"Beta_{ch}"] + 1e-12)
                feats[f"theta_alpha_ratio_{ch}"] = r[f"Theta_{ch}"] / (r[f"Alpha_{ch}"] + 1e-12)

            feats["frontal_alpha_asym"] = (
                np.log(r["Alpha_AF7"] + 1e-12) - np.log(r["Alpha_AF8"] + 1e-12)
            )
            rows.append(feats)

        return pd.DataFrame(rows)

    else:
        raise ValueError("Unknown schema (expected 'raw' or 'band').")


def choose_cv(y: pd.Series) -> StratifiedKFold:
    # Stratified 2–5 folds depending on minimum class count
    min_per = y.value_counts().min()
    n_splits = max(2, min(5, int(min_per)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ------- main -------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--dur-grid", default="30,60,120,180,240,300")
    ap.add_argument("--min-good", type=float, default=0.0,
                    help="(Reserved) Minimum fraction of good windows per participant.")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="(Reserved) Minimum usable duration in seconds.")
    ap.add_argument("--positive-regex", default=None,
                    help="Binary mapping regex for POS class (case-insensitive).")
    args = ap.parse_args()

    root = Path(".").resolve()
    man = (root / args.manifest).resolve()
    dfm = load_manifest(man)

    outdir = (root / "data/features/duration_test").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- 0) Collect window features per participant --------
    per_part = {}
    meta_rows = []

    for _, r in dfm.iterrows():
        pid = str(r["participant_id"])
        fid = str(r["file_id"])
        gl = str(r["group_label"])
        csv_path = (root / str(r["csv_path"])).resolve()

        if not csv_path.exists():
            continue

        df = load_csv(csv_path)
        schema = detect_schema(df.columns)
        fs = int(r["sr_expected"])

        try:
            W = extract_windows(df, fs, schema)  # n_win x n_feat
        except Exception as e:
            print(f"[WARN] Skipping file_id={fid} due to error: {e}")
            continue

        if W.empty:
            continue

        per_part[fid] = (pid, gl, W, fs)
        meta_rows.append({
            "file_id": fid,
            "participant_id": pid,
            "group_label": gl,
            "n_windows": len(W),
            "fs": fs,
        })

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(outdir / "windows_meta.csv", index=False)

    if not per_part:
        print("No participants with usable windows. Exiting.")
        return

    # Duration grid (seconds)
    grid = [int(x) for x in args.dur_grid.split(",")]

    # Feature set to track (keep small & meaningful)
    track_feats = [
        "theta_beta_ratio_AF7", "theta_beta_ratio_AF8",
        "theta_alpha_ratio_AF7", "theta_alpha_ratio_AF8",
        "theta_rel_AF7", "theta_rel_AF8",
        "frontal_alpha_asym",
    ]

    # -------- 1) Feature convergence per participant --------
    conv_rows = []
    for fid, (pid, gl, W, fs) in per_part.items():
        step_s = WIN_S * (1 - WIN_OVERLAP)  # e.g., 2s if 4s win & 50% overlap
        dur_per_win = step_s

        for T in grid:
            k = max(1, int(T / dur_per_win))
            Wsub = W.head(k)
            med = Wsub.median(numeric_only=True).to_dict()

            for f in track_feats:
                if f in med:
                    conv_rows.append({
                        "file_id": fid,
                        "participant_id": pid,
                        "group_label": gl,
                        "duration_s": T,
                        "feature": f,
                        "median": med[f],
                    })

    conv = pd.DataFrame(conv_rows)
    conv.to_csv(outdir / "feature_convergence.csv", index=False)

    # quick convergence plots (median vs duration)
    plot_dir = outdir / "feature_stability_plots"
    plot_dir.mkdir(exist_ok=True)

    for f in track_feats:
        sub = conv[conv["feature"] == f]
        if sub.empty:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for gl, gdf in sub.groupby("group_label"):
            agg = gdf.groupby("duration_s")["median"].median()
            ax.plot(agg.index, agg.values, marker="o", label=str(gl))

        ax.set_title(f"Convergence: {f}")
        ax.set_xlabel("Cumulative duration (s)")
        ax.set_ylabel("Median over windows")
        ax.legend()
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(plot_dir / f"convergence_{f}.png", dpi=150)
        plt.close(fig)

    # -------- 2) Split-half reliability --------
    sh_rows = []
    for fid, (pid, gl, W, fs) in per_part.items():
        n = len(W)
        if n < 6:
            continue

        h = n // 2
        A = W.iloc[:h]
        B = W.iloc[h:]

        mA = A.median(numeric_only=True)
        mB = B.median(numeric_only=True)

        for f in track_feats:
            if f in mA and f in mB:
                vA, vB = mA[f], mB[f]
                denom = (abs(vA) + abs(vB)) / 2 + 1e-12
                rel = 1.0 - (abs(vA - vB) / denom)   # 1 = perfect agreement
                sh_rows.append({
                    "file_id": fid,
                    "participant_id": pid,
                    "group_label": gl,
                    "feature": f,
                    "split_half_reliability": rel,
                })

    split_half = pd.DataFrame(sh_rows)
    split_half.to_csv(outdir / "split_half_reliability.csv", index=False)

    # -------- 3) Accuracy vs duration --------

    def build_table_at_T(T: int) -> pd.DataFrame:
        rows = []
        for fid, (pid, gl, W, fs) in per_part.items():
            step_s = WIN_S * (1 - WIN_OVERLAP)
            k = max(1, int(T / step_s))
            Wsub = W.head(k)
            med = Wsub.median(numeric_only=True)

            row = {
                "file_id": fid,
                "participant_id": pid,
                "group_label": gl,
            }
            for f in track_feats:
                row[f] = med.get(f, np.nan)
            rows.append(row)

        dfT = pd.DataFrame(rows).dropna()
        return dfT

    def map_labels(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        if args.positive_regex:
            mask = s.str.contains(args.positive_regex, case=False, regex=True, na=False)
            return mask.map({True: "POS", False: "NEG"})
        return s

    acc_rows = []
    for T in grid:
        dfT = build_table_at_T(T)
        if dfT.empty:
            acc_rows.append({
                "duration_s": T,
                "balanced_accuracy": np.nan,
                "n": 0,
            })
            continue

        y = map_labels(dfT["group_label"])

        # Drop super-rare classes (count < 2) *for this analysis only*
        vc = y.value_counts()
        rare = vc[vc < 2].index
        if len(rare) > 0:
            keep_mask = ~y.isin(rare)
            y = y[keep_mask]
            dfT = dfT[keep_mask]

        if y.nunique() < 2:
            acc_rows.append({
                "duration_s": T,
                "balanced_accuracy": np.nan,
                "n": len(dfT),
            })
            continue

        X = dfT[track_feats].copy()
        X = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X),
            columns=track_feats,
        )
        X = pd.DataFrame(
            RobustScaler().fit_transform(X),
            columns=track_feats,
        )

        clf = make_pipeline(
            SVC(kernel="linear", class_weight="balanced", random_state=42)
        )
        cv = choose_cv(y)
        yhat = cross_val_predict(clf, X, y, cv=cv)
        bacc = balanced_accuracy_score(y, yhat)

        acc_rows.append({
            "duration_s": T,
            "balanced_accuracy": bacc,
            "n": len(dfT),
        })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(outdir / "accuracy_vs_duration.csv", index=False)

    # plot accuracy vs duration
    if not acc_df.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(acc_df["duration_s"], acc_df["balanced_accuracy"], marker="o")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Cumulative duration (s)")
        ax.set_ylabel("Balanced accuracy (CV)")
        ax.set_title("Accuracy vs duration")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(outdir / "accuracy_vs_duration.png", dpi=160)
        plt.close(fig)

    print(f"\n✅ Wrote results to: {outdir}")
    print(" - windows_meta.csv")
    print(" - feature_convergence.csv, split_half_reliability.csv, accuracy_vs_duration.csv")
    print(" - plots: feature_stability_plots/, accuracy_vs_duration.png")


if __name__ == "__main__":
    main()
