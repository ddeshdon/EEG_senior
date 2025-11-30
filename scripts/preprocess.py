#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch

# ======================== CONFIG ========================

# Required manifest columns
REQUIRED = ["participant_id","group_label","csv_path","file_id","sr_expected","channels_expected"]

# canonical channels
CHANS = ["TP9","AF7","AF8","TP10"]

# raw aliases (prefer RAW_* if present)
ALIASES = {
    "TP9":  ["RAW_TP9","TP9","EEG.TP9","tp9","TP9 "],
    "AF7":  ["RAW_AF7","AF7","EEG.AF7","af7","AF7 "],
    "AF8":  ["RAW_AF8","AF8","EEG.AF8","af8","AF8 "],
    "TP10": ["RAW_TP10","TP10","EEG.TP10","tp10","TP10 "],
}

# frequency bands
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}

# duration + preprocessing params
MIN_DURATION_S = 180.0   # warn if shorter than this, but continue
BANDPASS = (1,45)        # Hz
NOTCH_HZ  = 50           # Hz
WIN_S = 4.0
WIN_OVERLAP = 0.5

# ======================== HELPERS ========================

def sha1_of_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def butter_bandpass(low, high, fs, order=4):
    from scipy.signal import butter
    return butter(order, [low/(fs/2), high/(fs/2)], btype="band")

def apply_filters(x, fs):
    b,a = butter_bandpass(*BANDPASS, fs, order=4)
    xf = filtfilt(b,a,x,axis=0)
    b,a = iirnotch(NOTCH_HZ, 30, fs)
    xf = filtfilt(b,a,xf,axis=0)
    return xf

def window_indices(n, fs, win_s=WIN_S, overlap=WIN_OVERLAP):
    w = int(win_s*fs); step = max(1, int(w*(1-overlap)))
    return [(s, s+w) for s in range(0, n-w+1, step)]

def bandpower_welch(sig, fs, fmin, fmax):
    f, Pxx = welch(sig, fs=fs, window="hamming", nperseg=int(fs), noverlap=int(fs*0.5))
    idx = (f>=fmin) & (f<fmax)
    return float(np.trapz(Pxx[idx], f[idx]))

def hjorth(sig):
    d1 = np.diff(sig); d2 = np.diff(d1)
    v0 = np.var(sig); v1 = np.var(d1); v2 = np.var(d2)
    act = v0
    mob = np.sqrt(v1/v0) if v0>0 else 0.0
    comp = np.sqrt(v2/v1)/mob if (v1>0 and mob>0) else 0.0
    return float(act), float(mob), float(comp)

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def parse_timestamp_series(ts_col: pd.Series) -> np.ndarray:
    # numeric first (sec or ms)
    try:
        t = pd.to_numeric(ts_col, errors="coerce").to_numpy(float)
        if np.isfinite(t).sum() >= 10:
            rng = float(np.nanmax(t) - np.nanmin(t))
            if rng > 1e6:  # likely ms
                t = t / 1000.0
            return t
    except Exception:
        pass
    # try datetime → seconds
    t = pd.to_datetime(ts_col, errors="coerce").astype("int64")/1e9
    return t.to_numpy(float)

def infer_fs_and_duration(df: pd.DataFrame):
    ts_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("timestamp","time","ts"):
            ts_col = c; break
    if ts_col is None:
        return None, None
    t = parse_timestamp_series(df[ts_col])
    t = t[np.isfinite(t)]
    if t.size < 10:
        return None, None
    t = np.unique(t)
    duration_s = float(t[-1] - t[0]) if t.size >= 2 else None
    dts = np.diff(t)
    dts = dts[(dts > 0) & np.isfinite(dts)]
    fs = float(np.median(1.0/dts)) if dts.size >= 10 else None
    return fs, duration_s

def detect_schema(cols):
    cl = [c.lower() for c in cols]
    has_raw  = all(name in cl for name in ["raw_tp9","raw_af7","raw_af8","raw_tp10"])
    has_band = any(re.match(r"(delta|theta|alpha|beta|gamma)_(tp9|af7|af8|tp10)$", c, re.I) for c in cols)
    if has_raw:  return "raw"
    if has_band: return "band"
    return "unknown"

def map_raw(df: pd.DataFrame) -> pd.DataFrame:
    # drop common non-EEG helpers
    df = df.drop(columns=[c for c in df.columns if c.strip().lower() in ("timestamp","elements","battery")], errors="ignore")
    colmap={}
    for canon, cands in ALIASES.items():
        found = next((c for c in cands if c in df.columns), None)
        if not found:
            raise ValueError(f"Missing raw channel '{canon}'. Got: {list(df.columns)[:14]} ...")
        colmap[canon] = found
    out = df[[colmap[ch] for ch in CHANS]].copy()
    out.columns = CHANS
    for c in CHANS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any").reset_index(drop=True)
    return out

# ======================== EXTRACTORS ========================

def extract_from_raw(df_raw: pd.DataFrame, fs_used: int, duration_hint=None):
    X = df_raw[CHANS].to_numpy(float)
    dur = float(duration_hint) if duration_hint is not None else (len(X)/fs_used)
    if dur < MIN_DURATION_S:
        print(f"   ! warning: short recording ({dur:.1f}s < {MIN_DURATION_S:.0f}s); continuing")

    X = X - np.nanmean(X, axis=0)
    Xf = apply_filters(X, fs_used)
    wins = window_indices(len(Xf), fs_used, WIN_S, WIN_OVERLAP)

    good, perwin = [], []
    for s,e in wins:
        W = Xf[s:e,:]
        amp_bad = (np.max(np.abs(W), axis=0) > 150).any()
        zW = (W - W.mean(0)) / (W.std(0)+1e-8)
        z_bad = (np.abs(zW) > 5).any()
        rel_high = []
        for ci in range(W.shape[1]):
            tot = bandpower_welch(W[:,ci], fs_used, 1,45)
            hi  = bandpower_welch(W[:,ci], fs_used, 35,45)
            rel_high.append(hi/(tot+1e-12))
        high_bad = (np.array(rel_high) > 0.5).any()
        ok = not (amp_bad or z_bad or high_bad)
        good.append(ok)
        if not ok:
            perwin.append(None); continue

        f={}
        for ci,ch in enumerate(CHANS):
            tot = bandpower_welch(W[:,ci], fs_used, 1,45)
            bp  = {n: bandpower_welch(W[:,ci], fs_used, *rng) for n,rng in BANDS.items()}
            for n,v in bp.items():
                f[f"{n}_abs_{ch}"] = v
                f[f"{n}_rel_{ch}"] = v/(tot+1e-12)
            f[f"theta_beta_ratio_{ch}"]  = bp["theta"]/(bp["beta"]+1e-12)
            f[f"theta_alpha_ratio_{ch}"] = bp["theta"]/(bp["alpha"]+1e-12)
            act,mob,comp = hjorth(W[:,ci])
            f[f"hjorth_activity_{ch}"]   = act
            f[f"hjorth_mobility_{ch}"]   = mob
            f[f"hjorth_complexity_{ch}"] = comp
        f["frontal_alpha_asym"] = np.log(f["alpha_abs_AF7"]+1e-12) - np.log(f["alpha_abs_AF8"]+1e-12)
        perwin.append(f)

    good = np.array(good, bool); good_frac = float(good.mean())
    keys = sorted({k for d in perwin if d for k in d})
    med, iqr = {}, {}
    for k in keys:
        vals = np.array([d[k] for d in perwin if d is not None], float)
        med[k] = float(np.nanmedian(vals)) if vals.size else np.nan
        iqr[k] = float(np.nanpercentile(vals,75)-np.nanpercentile(vals,25)) if vals.size else np.nan
    return dur, good_frac, med, iqr

def extract_from_band(df: pd.DataFrame):
    # keep only band columns and coerce to numeric
    band_cols = [c for c in df.columns if re.match(r"^(Delta|Theta|Alpha|Beta|Gamma)_(TP9|AF7|AF8|TP10)$", c, re.I)]
    if not band_cols:
        raise ValueError("Band schema detected but no Delta/Theta/Alpha/Beta/Gamma columns found.")
    df2 = df[band_cols].copy()
    for c in band_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(how="any").reset_index(drop=True)
    if len(df2) < 50:
        print(f"   ! warning: only {len(df2)} band rows; continuing")

    chans = CHANS; bands = ["Delta","Theta","Alpha","Beta","Gamma"]
    rel_cols=[]
    for ch in chans:
        tot = df2[[f"{b}_{ch}" for b in bands]].sum(axis=1) + 1e-12
        for b in bands:
            name = f"{b.lower()}_rel_{ch}"
            df2[name] = df2[f"{b}_{ch}"] / tot
            rel_cols.append(name)

    df2["theta_beta_ratio_AF7"]  = df2["Theta_AF7"]/(df2["Beta_AF7"]+1e-12)
    df2["theta_beta_ratio_AF8"]  = df2["Theta_AF8"]/(df2["Beta_AF8"]+1e-12)
    df2["theta_alpha_ratio_AF7"] = df2["Theta_AF7"]/(df2["Alpha_AF7"]+1e-12)
    df2["theta_alpha_ratio_AF8"] = df2["Theta_AF8"]/(df2["Alpha_AF8"]+1e-12)
    df2["frontal_alpha_asym"]    = np.log(df2["Alpha_AF7"]+1e-12) - np.log(df2["Alpha_AF8"]+1e-12)

    med, iqr = {}, {}
    # absolute bands with standard names
    for b in bands:
        for ch in chans:
            v = df2[f"{b}_{ch}"].to_numpy(float)
            med[f"{b.lower()}_abs_{ch}"] = float(np.nanmedian(v))
            iqr[f"{b.lower()}_abs_{ch}"] = float(np.nanpercentile(v,75)-np.nanpercentile(v,25))
    # relatives + ratios + asym
    for c in rel_cols + ["theta_beta_ratio_AF7","theta_beta_ratio_AF8","theta_alpha_ratio_AF7","theta_alpha_ratio_AF8","frontal_alpha_asym"]:
        v = df2[c].to_numpy(float)
        med[c] = float(np.nanmedian(v))
        iqr[c] = float(np.nanpercentile(v,75)-np.nanpercentile(v,25))
    # no precise duration / window quality in band schema
    return np.nan, 1.0, med, iqr

# ======================== MAIN ========================

def main(args):
    # resolve paths relative to project root
    root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()

    print(f"[info] reading manifest from: {manifest}")
    m = pd.read_csv(manifest)
    miss = [c for c in REQUIRED if c not in m.columns]
    if miss:
        raise SystemExit(f"Manifest missing columns: {miss}")

    # PASS-THROUGH LABELS: keep exactly what you wrote
    m["group_label_raw"] = m["group_label"]
    m["group_label"] = m["group_label"].astype(str).str.strip()

    feat_dir = Path(args.features_dir); int_dir = Path(args.interim_dir)
    if not feat_dir.is_absolute(): feat_dir = (root/feat_dir).resolve()
    if not int_dir.is_absolute():  int_dir  = (root/int_dir).resolve()
    feat_dir.mkdir(parents=True, exist_ok=True); int_dir.mkdir(parents=True, exist_ok=True)

    wide = []
    for _, row in m.iterrows():
        pid = str(row["participant_id"]); gl = row["group_label"]; gl_raw = row["group_label_raw"]
        fid = str(row["file_id"])
        sr  = int(row["sr_expected"])  # may be overridden by inference
        csv_path = Path(str(row["csv_path"]))
        if not csv_path.is_absolute(): csv_path = (root / csv_path).resolve()
        if not csv_path.exists():
            print(f"[WARN] {fid}: missing {csv_path}"); continue

        print(f"→ Processing {fid} ({pid}) from {csv_path} ...")

        try:
            df = load_csv(csv_path)
            # infer fs & duration from TimeStamp if present
            fs_inferred, dur_inferred = infer_fs_and_duration(df)
            if fs_inferred is not None:
                sr_used = int(round(fs_inferred))
                print(f"   · inferred fs ≈ {fs_inferred:.1f} Hz → using fs={sr_used}")
            else:
                sr_used = sr
                print(f"   · no TimeStamp-based fs; using manifest fs={sr_used}")

            # >>> NEW: skip files shorter than MIN_DURATION_S (3 minutes) <<<
            if dur_inferred is not None and dur_inferred < MIN_DURATION_S:
                print(f"   ! skipping {fid}: duration {dur_inferred:.1f}s < {MIN_DURATION_S:.0f}s (min required)")
                continue
            # <<< END NEW >>>

            # drop obvious non-EEG helpers
            df = df.drop(columns=[c for c in df.columns if c.strip().lower() in ("elements","battery")], errors="ignore")
            schema = detect_schema(df.columns)
            print(f"   · schema detected: {schema}")

            if schema == "raw":
                df_raw = map_raw(df)
                dur, good, med, iqr = extract_from_raw(df_raw, fs_used=sr_used, duration_hint=dur_inferred)
            elif schema == "band":
                dur, good, med, iqr = extract_from_band(df)
            else:
                raise ValueError(f"Unknown schema. Columns: {list(df.columns)[:12]} ...")

        except Exception as e:
            print(f"[ERR] {fid}: {e}")
            continue

        # write per-file feature + preproc summaries
        (feat_dir / f"{fid}.features.json").write_text(json.dumps({
            "file_id": fid,
            "participant_id": pid,
            "group_label": gl,          # YOUR raw label
            "group_label_raw": gl_raw,  # copy (same as above)
            "sr_hz_used": sr_used,
            "duration_s": dur,
            "good_window_frac": good,
            "features_median": med,
            "features_iqr": iqr
        }, indent=2, ensure_ascii=False))

        (int_dir / f"{fid}.preproc.json").write_text(json.dumps({
            "file_id": fid,
            "csv_path": str(csv_path),
            "csv_sha1": sha1_of_file(csv_path),
            "schema_used": schema,
            "fs_used_hz": sr_used,
            "fs_inferred_hz": fs_inferred,
            "duration_inferred_s": dur_inferred,
            "min_duration_required_s": MIN_DURATION_S,
            "bandpass_hz": BANDPASS,
            "notch_hz": NOTCH_HZ,
            "window_len_s": WIN_S,
            "overlap": WIN_OVERLAP,
            "artifact_rule": "amp>150uV OR |z|>5 OR rel(35-45)>0.5",
            "good_window_frac": good
        }, indent=2, ensure_ascii=False))

        # compact row for quick stats
        w = {
            "file_id": fid, "participant_id": pid,
            "group_label": gl, "duration_s": dur,
            "good_window_frac": good
        }
        for k in ("alpha_rel_AF7","alpha_rel_AF8","theta_rel_AF7","theta_rel_AF8",
                  "theta_beta_ratio_AF7","theta_beta_ratio_AF8","frontal_alpha_asym"):
            if k in med:
                w[f"{k}__median"] = med[k]
                w[f"{k}__iqr"] = iqr.get(k, np.nan)
        wide.append(w)

        print(f"   ✓ done. schema={schema}, good={good:.2f}, duration={('NA' if np.isnan(dur) else f'{dur:.1f}s')}")

    if wide:
        pd.DataFrame(wide).to_csv(feat_dir/"features_wide.csv", index=False)
        print(f"\n✅ Wrote {len(wide)} rows to {feat_dir/'features_wide.csv'}")
    else:
        print("\n[WARN] No rows processed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--interim-dir", default="data/interim")
    args = ap.parse_args()
    main(args)