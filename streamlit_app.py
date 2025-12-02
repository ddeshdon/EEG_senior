#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, welch

import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = Path("models/experienced_catboost_model.joblib")
TRAIN_WIDE_PATH = Path("data/features/features_wide.csv")
UNI_PATH = Path("data/features/diagnostics/univariate_binary.csv")

CHANS = ["TP9", "AF7", "AF8", "TP10"]
ALIASES = {
    "TP9":  ["RAW_TP9", "TP9", "EEG.TP9", "tp9", "TP9 "],
    "AF7":  ["RAW_AF7", "AF7", "EEG.AF7", "af7", "AF7 "],
    "AF8":  ["RAW_AF8", "AF8", "EEG.AF8", "af8", "AF8 "],
    "TP10": ["RAW_TP10", "TP10", "EEG.TP10", "tp10", "TP10 "],
}
BAND_PREFIXES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BANDPASS = (1, 45)
NOTCH_HZ = 50
WIN_S = 4.0
WIN_OVERLAP = 0.5
MIN_DURATION_S = 180.0
FS_DEFAULT = 256.0

st.set_page_config(
    page_title="Meditation EEG Classifier",
    page_icon="🧘‍♀️",
    layout="wide",
)

# ==========================================================
# LOAD MODEL + DATA
# ==========================================================

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)

    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    clf = bundle["classifier_calibrated"]
    feature_names = bundle["feature_names"]
    feature_selector_mask = np.array(bundle["feature_selector_mask"], bool)
    selected_features = bundle["selected_features"]
    label_mapping = bundle["label_mapping"]
    shap_global = bundle.get("shap_global_importance", {})

    # normalise SHAP format to dict
    if isinstance(shap_global, dict):
        shap_importance = shap_global
    else:
        shap_importance = {}
        for item in shap_global:
            if not isinstance(item, dict):
                continue
            feat = item.get("feature")
            val = item.get("importance", item.get("mean_abs_shap", None))
            if feat is not None and val is not None:
                shap_importance[feat] = val

    return (
        imputer,
        scaler,
        clf,
        feature_names,
        feature_selector_mask,
        selected_features,
        label_mapping,
        shap_importance,
    )


@st.cache_resource
def load_training_wide():
    if not TRAIN_WIDE_PATH.exists():
        return None
    df = pd.read_csv(TRAIN_WIDE_PATH)

    # map 4 labels → binary 0/1
    map_bin = {
        "beginner": 0,
        "never_practiced": 0,
        "trained_over_3_years": 1,
        "meditation_master_or_instructor": 1,
    }
    df = df[df["group_label"].isin(map_bin.keys())].copy()
    df["binary_label"] = df["group_label"].map(map_bin)
    return df


@st.cache_resource
def load_univariate():
    if not UNI_PATH.exists():
        return None
    df = pd.read_csv(UNI_PATH)
    if "cliffs_delta" in df.columns:
        df["abs_cliffs_delta"] = df["cliffs_delta"].abs()
    return df


(
    imputer,
    scaler,
    clf,
    feature_names,
    feature_selector_mask,
    selected_features,
    label_mapping,
    shap_importance,
) = load_model_bundle()

train_wide = load_training_wide()
uni_df = load_univariate()

# ==========================================================
# HELPERS: TIME, FILTERS, RAW MAPPING
# ==========================================================

def parse_timestamp_series(ts_col: pd.Series) -> np.ndarray:
    """Try numeric first, then datetime, return seconds."""
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


def infer_fs_duration(df: pd.DataFrame):
    ts_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("timestamp", "time", "ts"):
            ts_col = c
            break
    if ts_col is None:
        return FS_DEFAULT, len(df) / FS_DEFAULT

    t = parse_timestamp_series(df[ts_col])
    t = t[np.isfinite(t)]
    if t.size < 3:
        return FS_DEFAULT, len(df) / FS_DEFAULT

    t = np.unique(t)
    dur = float(t[-1] - t[0])
    dts = np.diff(t)
    dts = dts[dts > 0]
    if dts.size < 5:
        return FS_DEFAULT, dur
    fs = float(np.median(1.0 / dts))
    return fs, dur


def butter_bandpass(low, high, fs, order=4):
    return butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")


def apply_filters(x, fs):
    b, a = butter_bandpass(*BANDPASS, fs, order=4)
    x = filtfilt(b, a, x, axis=0)
    b, a = iirnotch(NOTCH_HZ, 30, fs)
    x = filtfilt(b, a, x, axis=0)
    return x


def bandpower(sig, fs, fmin, fmax):
    f, Pxx = welch(
        sig,
        fs=fs,
        window="hamming",
        nperseg=int(fs),
        noverlap=int(fs * 0.5),
    )
    idx = (f >= fmin) & (f < fmax)
    return float(np.trapz(Pxx[idx], f[idx]))


def hjorth(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = np.var(sig)
    v1 = np.var(d1)
    v2 = np.var(d2)
    mob = np.sqrt(v1 / v0) if v0 > 0 else 0.0
    comp = np.sqrt(v2 / v1) / mob if (v1 > 0 and mob > 0) else 0.0
    return float(v0), float(mob), float(comp)


def map_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[c for c in df.columns if c.strip().lower() in ("timestamp", "elements", "battery")],
        errors="ignore",
    )
    colmap = {}
    for canon, cands in ALIASES.items():
        found = next((c for c in cands if c in df.columns), None)
        if not found:
            raise ValueError(
                f"Missing raw channel '{canon}'. Columns available: {list(df.columns)[:10]}"
            )
        colmap[canon] = found
    out = df[[colmap[ch] for ch in CHANS]].copy()
    out.columns = CHANS
    for c in CHANS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any").reset_index(drop=True)
    return out

# ==========================================================
# FEATURE EXTRACTION (RAW / BAND TABLE)
# ==========================================================

def extract_raw_features(df: pd.DataFrame) -> pd.Series:
    """Raw Muse CSV → windowed features → median & IQR."""
    fs, dur = infer_fs_duration(df)
    df_raw = map_raw(df)
    X = df_raw[CHANS].to_numpy(float)

    if dur < MIN_DURATION_S:
        st.warning(
            f"⚠️ Short recording ({dur:.1f}s < {MIN_DURATION_S:.0f}s). "
            "Prediction may be less reliable."
        )

    X = X - np.nanmean(X, axis=0)
    Xf = apply_filters(X, fs)

    wlen = int(WIN_S * fs)
    step = max(1, int(wlen * (1.0 - WIN_OVERLAP)))
    if wlen >= len(Xf):
        wins = [(0, len(Xf))]
    else:
        wins = [(i, i + wlen) for i in range(0, len(Xf) - wlen + 1, step)]

    feats_per_win = []
    good = []

    for s, e in wins:
        W = Xf[s:e, :]

        # simple artifact rule
        amp_bad = (np.max(np.abs(W), axis=0) > 150).any()
        if amp_bad:
            feats_per_win.append(None)
            good.append(False)
            continue

        f = {}
        ch_bp = {}
        for ci, ch in enumerate(CHANS):
            sig = W[:, ci]
            tot = bandpower(sig, fs, 1, 45)
            bp = {b: bandpower(sig, fs, *rng) for b, rng in BANDS.items()}
            ch_bp[ch] = bp

            for b, v in bp.items():
                f[f"{b}_rel_{ch}"] = v / (tot + 1e-12)

            f[f"theta_beta_ratio_{ch}"] = bp["theta"] / (bp["beta"] + 1e-12)

            act, mob, comp = hjorth(sig)
            f[f"hjorth_activity_{ch}"] = act

        # frontal alpha asymmetry
        if "AF7" in ch_bp and "AF8" in ch_bp:
            f["frontal_alpha_asym"] = np.log(ch_bp["AF7"]["alpha"] + 1e-12) - np.log(
                ch_bp["AF8"]["alpha"] + 1e-12
            )

        feats_per_win.append(f)
        good.append(True)

    good = np.array(good, bool)
    good_frac = float(good.mean()) if good.size else 0.0

    keys = sorted({k for d in feats_per_win if d is not None for k in d.keys()})
    med, iqr = {}, {}
    for k in keys:
        vals = np.array([d[k] for d in feats_per_win if d is not None and k in d], float)
        if vals.size:
            med[k] = float(np.nanmedian(vals))
            iqr[k] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        else:
            med[k] = np.nan
            iqr[k] = np.nan

    out = {}
    for k, v in med.items():
        out[f"{k}__median"] = v
    for k, v in iqr.items():
        out[f"{k}__iqr"] = v
    out["duration_s"] = float(dur)
    out["good_window_frac"] = good_frac
    return pd.Series(out)


def extract_band_table_features(df: pd.DataFrame) -> pd.Series:
    """Band table (Delta_TP9, Theta_AF7, ...) → median & IQR per column."""
    fs, dur = infer_fs_duration(df)
    cols = [c for c in df.columns if c.lower() not in ("timestamp", "time", "ts")]
    med, iqr = {}, {}
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            med[c] = float(np.nanmedian(vals))
            iqr[c] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        else:
            med[c] = np.nan
            iqr[c] = np.nan

    out = {}
    for k, v in med.items():
        out[f"{k}__median"] = v
    for k, v in iqr.items():
        out[f"{k}__iqr"] = v
    out["duration_s"] = float(dur)
    out["good_window_frac"] = 1.0
    return pd.Series(out)

# ==========================================================
# MODE DETECTION & PREDICTION
# ==========================================================

def detect_mode(df: pd.DataFrame) -> str:
    cols = set(df.columns)

    # already feature row
    if len([c for c in selected_features if c in cols]) >= 3:
        return "features"

    # raw
    raw_cols = set(sum(ALIASES.values(), []))
    if len(cols.intersection(raw_cols)) > 0:
        return "raw"

    # band table
    for c in cols:
        if "_" in c:
            if c.split("_")[0] in BAND_PREFIXES:
                return "band_table"

    return "unknown"


def run_prediction(feature_row: pd.Series):
    # build full X with all feature_names in correct order
    full = {f: feature_row.get(f, np.nan) for f in feature_names}
    X = pd.DataFrame([full], columns=feature_names)

    X_imp = imputer.transform(X)
    X_scl = scaler.transform(X_imp)
    X_sel = X_scl[:, feature_selector_mask]

    proba = clf.predict_proba(X_sel)[0]
    pred_idx = int(np.argmax(proba))
    label = label_mapping.get(pred_idx, str(pred_idx))
    prob_dict = {label_mapping.get(i, str(i)): float(p) for i, p in enumerate(proba)}
    return label, prob_dict

# ==========================================================
# EXPLANATION HELPERS
# ==========================================================

def describe_feature_name(f: str) -> str:
    f_low = f.lower()
    desc = []
    if "alpha" in f_low:
        desc.append("alpha (8–13 Hz, relaxed wakefulness)")
    if "theta" in f_low:
        desc.append("theta (4–8 Hz, internal focus)")
    if "delta" in f_low:
        desc.append("delta (1–4 Hz, slow activity)")
    if "beta" in f_low:
        desc.append("beta (13–30 Hz, active thinking)")
    if "gamma" in f_low:
        desc.append("gamma (>30 Hz, high-frequency)")
    if "rel_" in f_low or "_rel_" in f_low:
        desc.append("relative power (proportion of total)")
    if "ratio" in f_low:
        desc.append("band ratio")
    if "frontal_alpha_asym" in f_low:
        desc.append("frontal alpha asymmetry (left–right balance)")
    if "af7" in f_low or "af8" in f_low:
        desc.append("frontal site")
    if "tp9" in f_low or "tp10" in f_low:
        desc.append("temporal site")
    return ", ".join(desc) if desc else f


def explain_prediction_for_top_features(feat_row: pd.Series, label: str, top_n: int = 5):
    """
    For the most important features, show:
    - your value
    - medians of each group
    - a 0–1 straight-line gauge where 0 = non-experienced, 1 = experienced.
    The percentage text is rendered *below* the chart to avoid clipping.
    """
    if train_wide is None:
        st.info("Training dataset not available → cannot show feature-level explanation.")
        return

    if label not in ("non_experienced", "experienced"):
        st.info("Explanation currently implemented only for binary labels.")
        return

    target_class = 1 if label == "experienced" else 0

    # sort features by global SHAP importance (most important first)
    imp_list = sorted(
        [(f, shap_importance.get(f, 0.0)) for f in selected_features],
        key=lambda x: x[1],
        reverse=True,
    )

    used = 0
    for f, _ in imp_list:
        if used >= top_n:
            break
        if f not in feat_row.index or f not in train_wide.columns:
            continue

        # your value
        val = float(feat_row[f])

        # medians of each group
        vals0 = train_wide.loc[train_wide["binary_label"] == 0, f].dropna()
        vals1 = train_wide.loc[train_wide["binary_label"] == 1, f].dropna()
        if vals0.empty or vals1.empty:
            continue

        med0 = float(vals0.median())
        med1 = float(vals1.median())

        # distances to each median
        d0 = abs(val - med0)
        d1 = abs(val - med1)

        # convert into "experienced score" between 0 and 1
        eps = 1e-9
        s0 = 1.0 / (d0 + eps)
        s1 = 1.0 / (d1 + eps)
        total = s0 + s1
        exp_score = s1 / total     # 0 = fully non-exp, 1 = fully exp
        closer_to = 0 if d0 <= d1 else 1

        st.markdown(f"### 🔍 {f}")
        st.write(f"**What this feature is about:** {describe_feature_name(f)}")

        st.markdown(
            f"- Your value: `{val:.4f}`  \n"
            f"- Median non-experienced (0): `{med0:.4f}`  \n"
            f"- Median experienced (1): `{med1:.4f}`"
        )

        # ---------- straight-line gauge (no internal text) ----------
        fig = go.Figure()

        # base line at y = 0.5
        fig.add_shape(
            type="line",
            x0=0,
            y0=0.5,
            x1=1,
            y1=0.5,
            line=dict(width=8),
        )

        # marker only
        fig.add_trace(
            go.Scatter(
                x=[exp_score],
                y=[0.5],
                mode="markers",
                marker=dict(size=18),
                showlegend=False,
            )
        )

        fig.update_xaxes(
            range=[0, 1],
            showgrid=False,
            tickvals=[0, 1],
            ticktext=["0 = non-experienced", "1 = experienced"],
            title_text="Leaning toward which group?",
        )
        fig.update_yaxes(visible=False, range=[0, 1])

        fig.update_layout(
            height=90,
            margin=dict(l=30, r=30, t=10, b=25),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # percentage text BELOW the chart (so Streamlit never clips it)
        st.markdown(
            f"<div style='text-align:center; font-size:0.9rem;'>"
            f"Gauge reading: <b>{exp_score*100:.0f}%</b> closer to <b>experienced (1)</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ---------- short text verdict ----------
        if closer_to == target_class:
            st.success(
                "For this feature, your value is **closer to the predicted class**, "
                "so it *supports* the prediction."
            )
        else:
            st.warning(
                "For this feature, your value is **closer to the opposite class**, "
                "so it partly *disagrees* with the prediction."
            )

        used += 1

# ==========================================================
# VISUALISATION: RAW / BAND TABLE (INTERACTIVE)
# ==========================================================

def plot_brainwaves_for_band_table(df: pd.DataFrame):
    """
    Plot one curve per band (Delta/Theta/Alpha/Beta/Gamma), averaging across channels.
    Interactive Plotly plot with zoom + range slider so small time fragments can be inspected.
    """
    ts_col = None
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "ts"):
            ts_col = c
            break

    if ts_col is not None:
        t = parse_timestamp_series(df[ts_col])
    else:
        t = np.arange(len(df))

    band_names = sorted({c.split("_")[0] for c in df.columns if "_" in c})
    band_names = [b for b in band_names if b.lower() in BANDS.keys()]
    if not band_names:
        st.info("No band columns (Delta_, Theta_, etc.) found for plotting.")
        return

    rows = []
    for band in band_names:
        band_cols = [c for c in df.columns if c.startswith(band + "_") or c == band]
        vals = pd.to_numeric(df[band_cols], errors="coerce").to_numpy(float)

        if vals.ndim == 1:
            mean_vals = vals
        else:
            mean_vals = np.nanmean(vals, axis=1)

        mean_vals = (mean_vals - np.nanmean(mean_vals)) / (np.nanstd(mean_vals) + 1e-12)

        for ti, v in zip(t, mean_vals):
            rows.append({"time": ti, "band": band.capitalize(), "value": v})

    plot_df = pd.DataFrame(rows)

    fig = px.line(
        plot_df,
        x="time",
        y="value",
        color="band",
        title="Absolute brain waves (per band, averaged across channels)",
        labels={"time": "Time", "value": "Relative power (z-score)", "band": "Band"},
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_brainwaves_from_raw(df: pd.DataFrame):
    """
    From raw Muse EEG compute band power time series and plot one curve per band.
    """
    fs, _ = infer_fs_duration(df)
    df_raw = map_raw(df)
    X = df_raw[CHANS].to_numpy(float)

    X = X - np.nanmean(X, axis=0)
    Xf = apply_filters(X, fs)

    win_s = 1.0
    step_s = 0.5
    wlen = int(win_s * fs)
    step = max(1, int(step_s * fs))
    if wlen >= len(Xf):
        wins = [(0, len(Xf))]
    else:
        wins = [(i, i + wlen) for i in range(0, len(Xf) - wlen + 1, step)]

    times = []
    band_series = {b: [] for b in BANDS.keys()}

    for s, e in wins:
        W = Xf[s:e, :]
        t_center = (s + e) / (2 * fs)
        times.append(t_center)

        for band, (fmin, fmax) in BANDS.items():
            bp_ch = []
            for ci in range(W.shape[1]):
                bp_ch.append(bandpower(W[:, ci], fs, fmin, fmax))
            band_series[band].append(float(np.nanmean(bp_ch)))

    rows = []
    t = np.array(times)
    for band, vals in band_series.items():
        vals = np.array(vals, dtype=float)
        if vals.size == 0 or np.all(~np.isfinite(vals)):
            continue

        vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)
        for ti, v in zip(t, vals):
            rows.append({"time": ti, "band": band.capitalize(), "value": v})

    if not rows:
        st.info("Not enough data to compute band powers for plotting.")
        return

    plot_df = pd.DataFrame(rows)

    fig = px.line(
        plot_df,
        x="time",
        y="value",
        color="band",
        title="Absolute brain waves (TP9 / AF7 / AF8 / TP10 → per-band power)",
        labels={"time": "Time (s)", "value": "Relative power (z-score)", "band": "Band"},
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGES
# ==========================================================

def page_predict():
    st.title("🧠 Meditation EEG Classifier")

    st.markdown(
        "Upload an EEG recording to estimate whether the pattern looks more like "
        "**non-experienced** or **experienced** meditators.\n\n"
        "Supported formats:\n"
        "- Raw Muse CSV (TP9 / AF7 / AF8 / TP10)\n"
        "- Band-power table (e.g., `Delta_TP9`, `Theta_AF7`, ...)\n"
        "- Feature row with the same columns as the trained model\n"
    )

    uploaded = st.file_uploader("Upload EEG CSV file", type=["csv"])

    if not uploaded:
        st.info("⬆️ Upload a raw Muse CSV, band-power table, or feature row to start.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    mode = detect_mode(df)

    st.subheader("Brainwave signal preview")
    st.caption("Interactive plot: drag to zoom, use the range slider to inspect small time fragments.")

    if mode == "band_table":
        plot_brainwaves_for_band_table(df)
    elif mode == "raw":
        plot_brainwaves_from_raw(df)
    elif mode == "features":
        st.info("File already contains features; skipping raw signal plot.")
    else:
        st.warning("Unrecognised format for plotting.")

    # feature extraction
    if mode == "raw":
        st.info("Detected raw Muse EEG → extracting features…")
        try:
            feat_row = extract_raw_features(df)
        except Exception as e:
            st.error(f"Error while extracting features: {e}")
            return
    elif mode == "band_table":
        st.info("Detected band-power table → summarising features…")
        feat_row = extract_band_table_features(df)
    elif mode == "features":
        st.info("Detected feature table → using first row.")
        feat_row = df.iloc[0]
    else:
        st.error(
            "Could not recognise this CSV.\n\n"
            "- No raw channels (RAW_TP9, RAW_AF7, ...),\n"
            "- No band columns like Delta_TP9, Theta_AF7, ...,\n"
            "- No known feature columns."
        )
        return

    # prediction
    label, prob_dict = run_prediction(feat_row)

    st.subheader("Prediction")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"**Predicted class:** `{label}`  \n"
            f"Model confidence for this class: `{prob_dict.get(label, 0.0):.3f}`"
        )
    with col2:
        prob_df = pd.DataFrame(
            {"class": list(prob_dict.keys()), "probability": list(prob_dict.values())}
        ).sort_values("probability", ascending=False)

        fig = px.bar(
            prob_df,
            x="class",
            y="probability",
            title="Class probabilities",
            labels={"class": "Class", "probability": "Predicted probability"},
        )
        fig.update_layout(
            xaxis_tickangle=0,
            yaxis_range=[0, 1],
            height=250,
            margin=dict(t=40, b=40, l=40, r=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why did the model predict this class?")
    st.markdown(
        "For the most important EEG features, we compare **your value** with the "
        "median of each group and show a simple **straight-line gauge**:  \n"
        "0 = non-experienced, 1 = experienced. The marker shows which side your "
        "feature leans toward."
    )

    explain_prediction_for_top_features(feat_row, label, top_n=5)

    # --------------------------------------------------------------
    # Optional analysis sections (for presentation / experts)
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Optional: model & feature analysis (for interpretation)")

    with st.expander("Global model feature importance (SHAP)", expanded=False):
        page_feature_summary(embedded=True)

    with st.expander("Univariate group differences (Cliff's δ, Mann–Whitney)", expanded=False):
        page_univariate(embedded=True)

# ==========================================================
# ANALYSIS SECTIONS (EMBEDDABLE)
# ==========================================================

def page_feature_summary(embedded: bool = False):
    if not embedded:
        st.title("📊 Model Feature Importance (SHAP)")

    if train_wide is None:
        st.error("Training dataset not found. Cannot show summary.")
        return

    st.markdown(
        "This section shows **global feature importance** from the CatBoost model "
        "(SHAP values averaged across all samples) and how these features "
        "are distributed in non-experienced vs experienced meditators."
    )

    # table of SHAP importance
    imp_df = pd.DataFrame(
        [
            {
                "feature": f,
                "importance": shap_importance.get(f, np.nan),
                "description": describe_feature_name(f),
            }
            for f in selected_features
        ]
    )
    imp_df = imp_df.sort_values("importance", ascending=False)
    st.subheader("Global model-based importance (SHAP)")
    st.dataframe(imp_df, use_container_width=True)

    # boxplots for top N
    top_n = st.slider(
        "Number of top features to plot (by SHAP importance)",
        3,
        min(10, len(imp_df)),
        6,
        key="shap_top_n",
    )

    top_feats = imp_df["feature"].iloc[:top_n].tolist()
    for f in top_feats:
        if f not in train_wide.columns:
            continue

        fig, ax = plt.subplots(figsize=(6, 3))
        train_wide.boxplot(column=f, by="binary_label", ax=ax)
        ax.set_title(f"{f} — distribution by class")
        ax.set_xlabel("Class (0 = non-experienced, 1 = experienced)")
        ax.set_ylabel("Feature value")
        plt.suptitle("")
        st.pyplot(fig)

        med0 = train_wide.loc[train_wide["binary_label"] == 0, f].median()
        med1 = train_wide.loc[train_wide["binary_label"] == 1, f].median()

        st.markdown(
            f"- Median (non-experienced, 0): `{med0:.4f}`  \n"
            f"- Median (experienced, 1): `{med1:.4f}`  \n"
            f"- Larger values in class "
            f"{'1 (experienced)' if med1 > med0 else '0 (non-experienced)'} "
            f"indicate this feature is characteristic of that group."
        )


def page_univariate(embedded: bool = False):
    if not embedded:
        st.title("📈 Univariate Feature Analysis (Cliff's δ, Mann–Whitney)")

    if uni_df is None:
        st.error(
            "Univariate CSV not found.\n\n"
            f"Expected at: `{UNI_PATH}`"
        )
        return

    if train_wide is None:
        st.error(
            "Training dataset (`features_wide.csv`) not found.\n\n"
            "Needed for boxplots by class."
        )
        return

    st.markdown(
        "This section summarises **univariate group differences** between:\n"
        "- **NEG** (non-experienced, label 0) and  \n"
        "- **POS** (experienced, label 1)\n\n"
        "Statistics shown:\n"
        "- Cliff's delta (effect size)  \n"
        "- Mann–Whitney p-value  \n"
        "- `med_NEG` / `med_POS` (group medians)\n"
    )

    # sort by absolute effect size
    uni_sorted = uni_df.sort_values(
        "abs_cliffs_delta" if "abs_cliffs_delta" in uni_df.columns else "cliffs_delta",
        ascending=False,
    )

    st.subheader("Univariate statistics table")
    st.dataframe(uni_sorted, use_container_width=True)

    # ------- Boxplots for top K features -------
    st.subheader("Top features by effect size (boxplots + interpretation)")

    top_k = st.slider(
        "Number of top-effect features to plot",
        3,
        min(10, len(uni_sorted)),
        5,
        key="uni_top_k",
    )

    # simple helper for effect size label
    def effect_size_label(delta_abs: float) -> str:
        if delta_abs < 0.147:
            return "negligible"
        elif delta_abs < 0.33:
            return "small"
        elif delta_abs < 0.474:
            return "medium"
        else:
            return "large"

    top_rows = uni_sorted.head(top_k)

    for _, row in top_rows.iterrows():
        f = row["feature"]
        if f not in train_wide.columns:
            st.info(f"Feature `{f}` not found in training wide table; skipping boxplot.")
            continue

        cd = float(row["cliffs_delta"])
        p = float(row["p_mannwhitney"])
        med_neg = float(row["med_NEG"])
        med_pos = float(row["med_POS"])
        delta_abs = abs(cd)
        es_label = effect_size_label(delta_abs)

        st.markdown(f"### 🔍 {f}")
        st.write(f"**What this feature is about:** {describe_feature_name(f)}")

        fig, ax = plt.subplots(figsize=(6, 3))
        train_wide.boxplot(column=f, by="binary_label", ax=ax)
        ax.set_title(f"{f} — distribution by class (NEG=0, POS=1)")
        ax.set_xlabel("Class (0 = non-experienced, 1 = experienced)")
        ax.set_ylabel("Feature value")
        plt.suptitle("")
        st.pyplot(fig)

        st.markdown(
            f"- Cliff's δ: `{cd:.2f}` (**{es_label} effect**)  \n"
            f"- p (Mann–Whitney): `{p:.3f}`  \n"
            f"- Median NEG (0): `{med_neg:.4f}`  \n"
            f"- Median POS (1): `{med_pos:.4f}`"
        )

        if med_pos > med_neg:
            higher_group = "experienced (POS, label 1)"
        elif med_pos < med_neg:
            higher_group = "non-experienced (NEG, label 0)"
        else:
            higher_group = "both groups equally"

        direction = "higher in POS (experienced)" if med_pos > med_neg else (
            "higher in NEG (non-experienced)" if med_pos < med_neg else "similar in both groups"
        )

        st.markdown(
            f"**Interpretation:**  \n"
            f"- The effect size is **{es_label}** (|δ| = {delta_abs:.2f}), meaning the distributions are "
            f"{'clearly' if es_label in ['medium', 'large'] else 'modestly' if es_label == 'small' else 'only slightly'} "
            f"different between groups.  \n"
            f"- This feature tends to be **{direction}**, indicating it is more characteristic of "
            f"**{higher_group}**.  \n"
            f"- A lower p-value (e.g., < 0.05) suggests this difference is statistically unlikely to be due to chance."
        )

        st.markdown("---")

# ==========================================================
# MAIN
# ==========================================================

def main():
    st.sidebar.title("Meditation EEG Classifier")
    st.sidebar.markdown(
        "1. Upload an EEG CSV.\n"
        "2. Inspect the brainwave preview.\n"
        "3. See the predicted meditation experience class.\n\n"
        "_Advanced analysis (SHAP & univariate stats) is available as optional sections "
        "below the prediction for expert users and for your presentation._"
    )

    page_predict()


if __name__ == "__main__":
    main()
