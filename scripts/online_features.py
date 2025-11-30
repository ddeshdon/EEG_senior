# scripts/online_features.py

import io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch

# ==== copy these from preprocess.py so they match exactly ====

CHANS = ["TP9", "AF7", "AF8", "TP10"]

ALIASES = {
    "TP9":  ["RAW_TP9","TP9","EEG.TP9","tp9","TP9 "],
    "AF7":  ["RAW_AF7","AF7","EEG.AF7","af7","AF7 "],
    "AF8":  ["RAW_AF8","AF8","EEG.AF8","af8","AF8 "],
    "TP10": ["RAW_TP10","TP10","EEG.TP10","tp10","TP10 "],
}

BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}

MIN_DURATION_S = 180.0
BANDPASS = (1,45)
NOTCH_HZ = 50
WIN_S = 4.0
WIN_OVERLAP = 0.5


def parse_timestamp_series(ts_col: pd.Series) -> np.ndarray:
    try:
        t = pd.to_numeric(ts_col, errors="coerce").to_numpy(float)
        if np.isfinite(t).sum() >= 10:
            rng = float(np.nanmax(t) - np.nanmin(t))
            if rng > 1e6:  # likely ms
                t = t / 1000.0
            return t
    except Exception:
        pass
    t = pd.to_datetime(ts_col, errors="coerce").astype("int64") / 1e9
    return t.to_numpy(float)


def infer_fs_and_duration(df: pd.DataFrame):
    ts_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("timestamp","time","ts"):
            ts_col = c
            break
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


def butter_bandpass(low, high, fs, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype="band")


def apply_filters(x, fs):
    b,a = butter_bandpass(*BANDPASS, fs, order=4)
    xf = filtfilt(b,a,x,axis=0)
    b,a = iirnotch(NOTCH_HZ, 30, fs)
    xf = filtfilt(b,a,xf,axis=0)
    return xf


def window_indices(n, fs, win_s=WIN_S, overlap=WIN_OVERLAP):
    w = int(win_s*fs)
    step = max(1, int(w*(1-overlap)))
    return [(s, s+w) for s in range(0, n-w+1, step)]


def bandpower_welch(sig, fs, fmin, fmax):
    f, Pxx = welch(sig, fs=fs, window="hamming",
                   nperseg=int(fs), noverlap=int(fs*0.5))
    idx = (f>=fmin) & (f<fmax)
    return float(np.trapz(Pxx[idx], f[idx]))


def hjorth(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = np.var(sig)
    v1 = np.var(d1)
    v2 = np.var(d2)
    act = v0
    mob = np.sqrt(v1/v0) if v0>0 else 0.0
    comp = np.sqrt(v2/v1)/mob if (v1>0 and mob>0) else 0.0
    return float(act), float(mob), float(comp)


def map_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns
                          if c.strip().lower() in ("timestamp","elements","battery")],
                 errors="ignore")
    colmap = {}
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


def extract_from_raw(df_raw: pd.DataFrame, fs_used: int, duration_hint=None):
    X = df_raw[CHANS].to_numpy(float)
    dur = float(duration_hint) if duration_hint is not None else (len(X)/fs_used)

    X = X - np.nanmean(X, axis=0)
    Xf = apply_filters(X, fs_used)
    wins = window_indices(len(Xf), fs_used, WIN_S, WIN_OVERLAP)

    good = []
    perwin = []

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
            perwin.append(None)
            continue

        f = {}
        for ci,ch in enumerate(CHANS):
            tot = bandpower_welch(W[:,ci], fs_used, 1,45)
            bp  = {n: bandpower_welch(W[:,ci], fs_used, *rng)
                   for n,rng in BANDS.items()}
            for n,v in bp.items():
                f[f"{n}_abs_{ch}"] = v
                f[f"{n}_rel_{ch}"] = v/(tot+1e-12)
            f[f"theta_beta_ratio_{ch}"]  = bp["theta"]/(bp["beta"]+1e-12)
            f[f"theta_alpha_ratio_{ch}"] = bp["theta"]/(bp["alpha"]+1e-12)
            act,mob,comp = hjorth(W[:,ci])
            f[f"hjorth_activity_{ch}"]   = act
            f[f"hjorth_mobility_{ch}"]   = mob
            f[f"hjorth_complexity_{ch}"] = comp
        f["frontal_alpha_asym"] = (
            np.log(f["alpha_abs_AF7"]+1e-12)
            - np.log(f["alpha_abs_AF8"]+1e-12)
        )
        perwin.append(f)

    good = np.array(good, bool)
    good_frac = float(good.mean())
    keys = sorted({k for d in perwin if d for k in d})
    med, iqr = {}, {}
    for k in keys:
        vals = np.array([d[k] for d in perwin if d is not None], float)
        med[k] = float(np.nanmedian(vals)) if vals.size else np.nan
        iqr[k] = float(
            np.nanpercentile(vals,75) - np.nanpercentile(vals,25)
        ) if vals.size else np.nan
    return dur, good_frac, med, iqr


def extract_features_for_model(uploaded_file) -> pd.DataFrame:
    """
    uploaded_file: streamlit UploadedFile or open file handle.
    Returns a 1-row DataFrame with the same columns as features_wide.csv.
    """
    # read CSV from file-like
    df = pd.read_csv(uploaded_file)

    fs_inferred, dur_inferred = infer_fs_and_duration(df)
    if fs_inferred is None:
        raise ValueError("Cannot infer sampling rate from TimeStamp")
    sr_used = int(round(fs_inferred))

    # use RAW schema
    df_raw = map_raw(df)
    dur, good_frac, med, iqr = extract_from_raw(df_raw, fs_used=sr_used,
                                                duration_hint=dur_inferred)

    if dur < MIN_DURATION_S:
        raise ValueError(f"Recording too short ({dur:.1f}s). Need ≥ {MIN_DURATION_S:.0f}s.")
    if good_frac < 0.3:
        raise ValueError(f"Too noisy (good_window_frac={good_frac:.2f} < 0.30).")

    # match the columns we used in features_wide.csv
    w = {
        "duration_s": dur,
        "good_window_frac": good_frac
    }
    # the summary subset you used when building features_wide
    for k in (
        "alpha_rel_AF7","alpha_rel_AF8",
        "theta_rel_AF7","theta_rel_AF8",
        "theta_beta_ratio_AF7","theta_beta_ratio_AF8",
        "frontal_alpha_asym"
    ):
        if k in med:
            w[f"{k}__median"] = med[k]
            w[f"{k}__iqr"] = iqr.get(k, np.nan)

    # return as 1-row DataFrame
    return pd.DataFrame([w])
