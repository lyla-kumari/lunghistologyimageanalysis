#!/usr/bin/env python3
"""
Lung histology — Alveolar airspaces + wall thickness (privacy-maximised Streamlit + CLI)

PRIVACY GOAL
- Do NOT persist uploads automatically.
- No Streamlit caching of uploaded bytes/arrays/masks/results.
- Default outputs are anonymized (random IDs), and metadata avoids filenames/paths.
- Streamlit: everything is processed in memory; exports happen only via download buttons.
- CLI: anonymize by default; use --keep_names if you explicitly want filenames.

Streamlit:
  streamlit run lungimageanalysis.py

CLI:
  python lungimageanalysis.py path/to/image.png --out_dir out
  python lungimageanalysis.py path/to/image.png --out_dir out --keep_names

Dependencies:
  pip install numpy pandas scipy scikit-image opencv-python pillow matplotlib streamlit
"""

import argparse
import json
import logging
import re
import sys
import uuid
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# =========================
# IO helpers
# =========================

def read_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    return arr.astype(np.uint8)


def read_rgb_from_bytes(b: bytes) -> np.ndarray:
    img = Image.open(BytesIO(b)).convert("RGB")
    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError("Empty uploaded image")
    return arr.astype(np.uint8)


def remove_exif_bytes(b: bytes) -> bytes:
    """Return image bytes with any EXIF/metadata removed by re-saving as PNG in memory.
    This prevents accidental leakage of identifying metadata when users upload images.
    """
    try:
        img = Image.open(BytesIO(b))
        rgb = img.convert("RGB")
        out = BytesIO()
        rgb.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        # If anything goes wrong, return the original bytes (best-effort)
        return b


def safe_stem(name: str) -> str:
    name = name or "image"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def save_rgb_png(rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def save_binary_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), img)


# =========================
# Segmentation
# =========================

def inpaint_specular(rgb: np.ndarray, val_quantile: float = 0.995, dilate_radius: int = 5) -> np.ndarray:
    """Detect very bright regions (specular highlights) and inpaint them."""
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_f)
    v = hsv[..., 2]
    try:
        v_lim = float(np.quantile(v, float(val_quantile)))
    except Exception:
        v_lim = 0.98

    mask = (v >= v_lim)
    if not mask.any():
        return rgb

    se = morphology.disk(int(max(1, dilate_radius)))
    mask = morphology.binary_dilation(mask, footprint=se)
    mask_u8 = (mask.astype(np.uint8) * 255)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def color_mask_from_rgb(
    rgb: np.ndarray,
    *,
    color_target: str = "purple",
    hue_min: Optional[float] = None,
    hue_max: Optional[float] = None,
    sat_thresh: float = 0.05,
) -> np.ndarray:
    """
    Hue-based pixel selection in HSV. Hue range is in [0,1].
    Wrap-around supported when hue_min > hue_max (e.g. pink 0.95..0.08).
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_f)
    h = hsv[..., 0]
    s = hsv[..., 1]

    ct = (color_target or "none").strip().lower()
    if ct in ("none", ""):
        return np.zeros(h.shape, dtype=bool)

    if ct == "purple":
        hmin = 0.65 if hue_min is None else float(hue_min)
        hmax = 0.90 if hue_max is None else float(hue_max)
    elif ct == "pink":
        hmin = 0.95 if hue_min is None else float(hue_min)
        hmax = 0.08 if hue_max is None else float(hue_max)
    elif ct == "custom":
        if hue_min is None or hue_max is None:
            return np.zeros(h.shape, dtype=bool)
        hmin, hmax = float(hue_min), float(hue_max)
    else:
        return np.zeros(h.shape, dtype=bool)

    if hmin <= hmax:
        hue_sel = (h >= hmin) & (h <= hmax)
    else:
        hue_sel = (h >= hmin) | (h <= hmax)

    return (hue_sel & (s > float(sat_thresh))).astype(bool)


def tissue_mask_advanced(
    rgb: np.ndarray,
    *,
    min_object_size: int = 150,
    close_radius: int = 2,
    open_radius: int = 1,
    gaussian_sigma: float = 1.0,
    inpaint: bool = False,
    val_quantile: float = 0.995,
    color_target: str = "purple",
    hue_min: Optional[float] = None,
    hue_max: Optional[float] = None,
    sat_thresh: float = 0.05,
    septa_max_px: float = 8.0,
) -> np.ndarray:
    """
    Segment tissue using either:
      - hue-targeted mask (purple/pink/custom), OR
      - saturation-based Otsu fallback if target='none'

    Then clean morphologically, and optionally apply thin-structure filter (EDT thickness <= septa_max_px)
    when color_target='purple' (septa emphasis).
    """
    rgb_proc = rgb.copy()
    if inpaint:
        rgb_proc = inpaint_specular(rgb_proc, val_quantile=val_quantile, dilate_radius=5)

    # saturation-based fallback
    rgb_f = rgb_proc.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_f)
    sat = hsv[..., 1]
    sat_smooth = ndi.gaussian_filter(sat, sigma=float(gaussian_sigma))
    try:
        thr = filters.threshold_otsu(sat_smooth)
    except Exception:
        thr = float(np.nanmedian(sat_smooth))
    base_mask = sat_smooth > thr

    ct = (color_target or "none").strip().lower()
    c_mask = color_mask_from_rgb(
        rgb_proc,
        color_target=ct,
        hue_min=hue_min,
        hue_max=hue_max,
        sat_thresh=sat_thresh,
    )

    combined = c_mask.copy() if ct not in ("none", "") else base_mask.copy()

    # morph cleanup
    combined = morphology.remove_small_objects(combined, min_size=int(max(1, min_object_size)))
    try:
        combined = morphology.remove_small_holes(combined, area_threshold=128)
    except Exception:
        pass
    combined = morphology.binary_closing(combined, footprint=morphology.disk(int(max(1, close_radius))))
    combined = morphology.binary_opening(combined, footprint=morphology.disk(int(max(1, open_radius))))
    combined = morphology.remove_small_objects(combined, min_size=int(max(1, min_object_size)))

    # thin-structure filter (septa emphasis)
    if ct == "purple" and septa_max_px is not None and float(septa_max_px) > 0:
        dist = ndi.distance_transform_edt(combined)
        thickness = 2.0 * dist
        combined = combined & (thickness <= float(septa_max_px))
        combined = morphology.remove_small_objects(combined, min_size=int(max(1, min_object_size)))

    return combined.astype(bool)


# =========================
# Measurement
# =========================

def measure_airspaces_and_wall_thickness(
    tissue_mask: np.ndarray,
    *,
    microns_per_pixel: Optional[float] = None,
    air_min_object_size: int = 16,
    wall_neigh_radius: int = 3,
    generate_skeleton: bool = False,
    skeleton_prune_px: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per airspace stats; wall thickness sampled from tissue pixels adjacent to each airspace.

    Returns:
      df, tissue_mask, skel, thickness_px_map
    """
    tissue_mask = tissue_mask.astype(bool)

    # thickness map for tissue pixels
    dist = ndi.distance_transform_edt(tissue_mask)
    thickness_px_map = 2.0 * dist

    # airspaces are inverse of tissue
    air_mask = (~tissue_mask).astype(bool)
    try:
        air_mask = morphology.remove_small_objects(air_mask, min_size=int(max(1, air_min_object_size)))
    except Exception:
        pass

    labeled, nlabels = ndi.label(air_mask)

    # optional skeleton of walls (for display only)
    skel = np.zeros_like(tissue_mask, dtype=bool)
    if generate_skeleton:
        try:
            wall_region = morphology.binary_dilation(
                air_mask, footprint=morphology.disk(int(max(1, wall_neigh_radius)))
            ) & tissue_mask
            skel = morphology.skeletonize(wall_region)
            if skeleton_prune_px and int(skeleton_prune_px) > 0:
                skel = morphology.remove_small_objects(skel.astype(bool), min_size=int(skeleton_prune_px))
        except Exception:
            skel = np.zeros_like(tissue_mask, dtype=bool)

    if nlabels == 0:
        cols = [
            "air_label", "area_px", "diameter_px", "centroid_y", "centroid_x",
            "wall_mean_thickness_px", "wall_median_thickness_px", "wall_p05_px", "wall_p95_px",
            "thickness_px",
        ]
        df = pd.DataFrame(columns=cols)
        if microns_per_pixel is not None:
            df["diameter_um"] = []
            df["area_um2"] = []
            df["thickness_um"] = []
        return df, tissue_mask, skel, thickness_px_map

    centroids = ndi.center_of_mass(air_mask, labeled, range(1, nlabels + 1))

    records = []
    rad = int(max(1, wall_neigh_radius))
    fp = morphology.disk(rad)

    for lbl in range(1, nlabels + 1):
        region = (labeled == lbl)
        area_px = int(region.sum())
        if area_px <= 0:
            continue

        diameter_px = float(np.sqrt(4.0 * area_px / np.pi))

        neigh = morphology.binary_dilation(region, footprint=fp)
        wall_region = neigh & tissue_mask
        vals = thickness_px_map[wall_region]

        if vals.size == 0:
            mean_th = median_th = p05 = p95 = np.nan
        else:
            mean_th = float(np.nanmean(vals))
            median_th = float(np.nanmedian(vals))
            p05 = float(np.nanpercentile(vals, 5))
            p95 = float(np.nanpercentile(vals, 95))

        cy, cx = centroids[lbl - 1] if centroids is not None else (np.nan, np.nan)

        records.append({
            "air_label": int(lbl),
            "area_px": area_px,
            "diameter_px": diameter_px,
            "centroid_y": float(cy) if np.isfinite(cy) else np.nan,
            "centroid_x": float(cx) if np.isfinite(cx) else np.nan,
            "wall_mean_thickness_px": mean_th,
            "wall_median_thickness_px": median_th,
            "wall_p05_px": p05,
            "wall_p95_px": p95,
            "thickness_px": median_th,  # canonical thickness per airspace
        })

    df = pd.DataFrame.from_records(records)

    if microns_per_pixel is not None:
        mpp = float(microns_per_pixel)
        df["area_um2"] = df["area_px"] * (mpp * mpp)
        df["diameter_um"] = df["diameter_px"] * mpp
        df["thickness_um"] = df["thickness_px"] * mpp

        df["wall_mean_thickness_um"] = df["wall_mean_thickness_px"] * mpp
        df["wall_median_thickness_um"] = df["wall_median_thickness_px"] * mpp
        df["wall_p05_um"] = df["wall_p05_px"] * mpp
        df["wall_p95_um"] = df["wall_p95_px"] * mpp

    return df, tissue_mask, skel, thickness_px_map


# =========================
# Statistics
# =========================

def _finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s[np.isfinite(s)]


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(int(seed))
    n = values.size
    stats = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = values[rng.integers(0, n, size=n)]
        stats[i] = float(stat_fn(samp))

    lo = float(np.quantile(stats, (1 - ci) / 2))
    hi = float(np.quantile(stats, 1 - (1 - ci) / 2))
    return lo, hi


def compute_stats_table(
    df: pd.DataFrame,
    *,
    columns: Tuple[str, ...],
    include_ci: bool = True,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = _finite_series(df, col)
        arr = s.to_numpy(dtype=float)

        if arr.size == 0:
            rows.append({
                "metric": col, "n": 0,
                "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan,
                "p05": np.nan, "p95": np.nan,
                "mean_ci_lo": np.nan, "mean_ci_hi": np.nan,
                "median_ci_lo": np.nan, "median_ci_hi": np.nan,
            })
            continue

        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        rec = {
            "metric": col,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "iqr": float(q3 - q1),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }

        if include_ci:
            mean_lo, mean_hi = bootstrap_ci(arr, np.mean, n_boot=int(n_boot), ci=float(ci), seed=int(seed))
            med_lo, med_hi = bootstrap_ci(arr, np.median, n_boot=int(n_boot), ci=float(ci), seed=int(seed) + 1)
            rec.update({
                "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
                "median_ci_lo": med_lo, "median_ci_hi": med_hi,
            })
        else:
            rec.update({
                "mean_ci_lo": np.nan, "mean_ci_hi": np.nan,
                "median_ci_lo": np.nan, "median_ci_hi": np.nan,
            })

        rows.append(rec)

    return pd.DataFrame(rows)


def corr_table(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    sub = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if sub.shape[1] == 0:
        return pd.DataFrame()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    return sub.corr(method="pearson")


# =========================
# Images / overlays
# =========================

def thickness_map_to_colormap_image(
    thickness_px_map: np.ndarray,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
) -> np.ndarray:
    if vmax is None:
        vmax = float(np.nanmax(thickness_px_map)) if np.isfinite(thickness_px_map).any() else 1.0
    denom = max(vmax - vmin, 1e-6)
    norm = np.clip((thickness_px_map - vmin) / denom, 0.0, 1.0)
    rgba = plt.get_cmap(cmap)(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap_on_rgb(rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if heatmap_rgb.shape != rgb.shape:
        heatmap_rgb = cv2.resize(heatmap_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_edge_overlay(rgb: np.ndarray, mask: np.ndarray, skel: Optional[np.ndarray] = None) -> np.ndarray:
    edges = cv2.Canny((mask.astype(np.uint8) * 255), 50, 150)
    overlay = rgb.copy()
    overlay[edges > 0] = [0, 255, 0]  # edges green
    if skel is not None and np.any(skel):
        overlay[skel.astype(bool)] = [255, 0, 0]  # skeleton red
    return overlay


def compute_alveoli_markers(
    air: np.ndarray,
    *,
    air_min_object_size: int = 16,
    marker_method: str = "local_maxima",
    marker_min_distance: int = 5,
    h_maxima_h: float = 1.0,
    dist_smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute markers for watershed separation of air regions.

    Returns (markers, dist_smooth) where markers is an integer-labelled marker image
    (0 background) and dist_smooth is the smoothed distance map used for watershed.
    """
    air = air.astype(bool)
    if not air.any():
        return np.zeros_like(air, dtype=int), np.zeros_like(air, dtype=float)

    dist = ndi.distance_transform_edt(air)
    sigma = float(dist_smooth_sigma) if dist_smooth_sigma is not None else 1.0
    dist_smooth = ndi.gaussian_filter(dist, sigma=max(0.0, sigma))

    mm = (marker_method or "local_maxima").strip().lower()
    try:
        if mm == "peak_local_max":
            try:
                pk = peak_local_max(dist_smooth, min_distance=int(max(1, marker_min_distance)), indices=False, labels=air)
            except TypeError:
                pk = peak_local_max(
                    dist_smooth,
                    min_distance=int(max(1, marker_min_distance)),
                    footprint=morphology.disk(int(max(1, marker_min_distance))),
                    labels=air,
                )
            markers_bool = pk.astype(bool)
            markers, _ = ndi.label(markers_bool)
        elif mm == "h_maxima":
            markers_bool = morphology.h_maxima(dist_smooth, h=float(max(0.0, h_maxima_h)))
            markers, _ = ndi.label(markers_bool)
        else:
            markers_bool = morphology.local_maxima(dist_smooth)
            markers_bool = markers_bool & air
            markers_bool = morphology.remove_small_objects(markers_bool, min_size=1)
            markers, _ = ndi.label(markers_bool)
    except Exception:
        markers = np.zeros_like(dist, dtype=int)

    return markers, dist_smooth


def alveoli_colored_overlay(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    air_min_object_size: int = 16,
    remove_border: bool = True,
    alpha: float = 0.55,
    seed: int = 0,
    marker_method: str = "local_maxima",
    marker_min_distance: int = 5,
    h_maxima_h: float = 1.0,
    dist_smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Create a coloured overlay of individual alveoli using watershed."""
    air = (~tissue_mask.astype(bool)).astype(bool)
    try:
        air = morphology.remove_small_objects(air, min_size=int(max(1, air_min_object_size)))
    except Exception:
        pass

    if not air.any():
        return rgb.copy(), 0

    markers, dist_smooth = compute_alveoli_markers(
        air,
        air_min_object_size=int(air_min_object_size),
        marker_method=str(marker_method),
        marker_min_distance=int(marker_min_distance),
        h_maxima_h=float(h_maxima_h),
        dist_smooth_sigma=float(dist_smooth_sigma),
    )

    try:
        labels = watershed(-dist_smooth, markers, mask=air)
    except Exception:
        labels, _ = ndi.label(air)

    if labels.max() > 0:
        counts = np.bincount(labels.ravel())
        small_ids = np.where(counts < int(max(1, air_min_object_size)))[0]
        small_ids = small_ids[small_ids != 0]
        if small_ids.size:
            labels[np.isin(labels, small_ids)] = 0
            labels, _ = ndi.label(labels > 0)

    n = int(labels.max())
    if n == 0:
        return rgb.copy(), 0

    if remove_border:
        border_ids = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_ids = border_ids[border_ids != 0]
        if border_ids.size:
            labels[np.isin(labels, border_ids)] = 0
            labels, _ = ndi.label(labels > 0)
            n = int(labels.max())
            if n == 0:
                return rgb.copy(), 0

    rng = np.random.default_rng(int(seed))
    palette = rng.integers(0, 256, size=(n + 1, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    color_img = palette[labels]

    overlay = alpha * color_img.astype(np.float32) + (1.0 - alpha) * rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    try:
        boundary = morphology.binary_dilation(labels > 0, footprint=morphology.disk(1)) ^ (labels > 0)
        overlay[boundary] = [255, 255, 255]
    except Exception:
        pass

    return overlay, int(n)


def plot_histogram_and_box_bytes(df: pd.DataFrame, col: str, bins: int = 60) -> bytes:
    s = _finite_series(df, col)
    arr = s.to_numpy(dtype=float)
    if arr.size == 0:
        return b""

    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6),
        gridspec_kw={"height_ratios": [1, 3]}
    )
    ax_box.boxplot(arr, vert=False)
    ax_box.set_yticks([])
    ax_hist.hist(arr, bins=int(bins), alpha=0.75)
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Count")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Streamlit app (privacy maximised)
# =========================

def streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Alveolar airspaces + wall thickness", layout="wide")

    st.title("Alveolar airspaces + wall thickness")
    st.caption(
        "Privacy-first: uploads are processed in memory. No server-side caching is used. "
        "Exports occur only when you download."
    )

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    # Always anonymize by default (privacy maximised)
    st.sidebar.header("Privacy")
    anonymize_outputs = st.sidebar.checkbox("Anonymize outputs (recommended)", value=True)
    # If anonymized -> random ID; if not -> safe stem from filename
    base = uuid.uuid4().hex if anonymize_outputs else safe_stem(getattr(uploaded, "name", "uploaded"))

    # -------- Sidebar controls --------
    st.sidebar.header("Calibration")
    microns_per_pixel = st.sidebar.number_input("Microns per pixel (0 = unknown)", value=0.0, format="%.6f")
    microns_per_pixel = None if microns_per_pixel == 0.0 else float(microns_per_pixel)

    st.sidebar.header("Segmentation")
    color_target = st.sidebar.selectbox("Color target", ["purple", "pink", "none", "custom"], index=0)
    with st.sidebar.expander("Hue tuning (used for purple/pink/custom preview)"):
        hue_min = st.slider("Hue min", 0.0, 1.0, 0.65, 0.01)
        hue_max = st.slider("Hue max", 0.0, 1.0, 0.90, 0.01)
        sat_thresh = st.slider("Saturation threshold", 0.0, 1.0, 0.05, 0.01)

    tissue_min_object_size = st.sidebar.number_input("tissue_min_object_size (px)", value=150, min_value=1, step=1)
    septa_max_px = st.sidebar.number_input("thin-structure max thickness (px)", value=8.0, min_value=0.1)
    inpaint = st.sidebar.checkbox("Inpaint specular highlights", value=False)

    st.sidebar.header("Measurement")
    air_min_object_size = st.sidebar.number_input("air_min_object_size (px)", value=16, min_value=1, step=1)
    wall_neigh_radius = st.sidebar.number_input("wall neighbourhood radius (px)", value=3, min_value=1, step=1)

    st.sidebar.header("Statistics")
    include_ci = st.sidebar.checkbox("Include bootstrap CIs (mean/median)", value=True)
    n_boot = st.sidebar.number_input("Bootstrap samples", value=2000, min_value=200, step=100)
    ci_level = st.sidebar.slider("CI level", 0.80, 0.99, 0.95, 0.01)
    stats_seed = st.sidebar.number_input("Stats seed", value=0, step=1)

    st.sidebar.header("Images (optional)")
    show_images = st.sidebar.checkbox("Enable image outputs", value=True)
    show_original = st.sidebar.checkbox("Show original image", value=True) if show_images else False
    show_color_preview = st.sidebar.checkbox("Show hue-target preview", value=True) if show_images else False
    show_overlay = st.sidebar.checkbox("Show edge overlay (edges green)", value=True) if show_images else False
    show_skeleton = st.sidebar.checkbox("Overlay skeleton (red)", value=False) if show_images else False
    show_heatmap = st.sidebar.checkbox("Show thickness heatmap overlay", value=True) if show_images else False
    show_masks = st.sidebar.checkbox("Show mask + skeleton images", value=True) if show_images else False
    show_alveoli_colored = st.sidebar.checkbox("Show coloured alveoli overlay", value=False) if show_images else False
    show_hists = st.sidebar.checkbox("Show histogram+boxplot", value=True) if show_images else False

    heat_alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.6, 0.05) if (show_images and show_heatmap) else 0.6
    alveoli_overlay_alpha = st.sidebar.slider("Alveoli overlay alpha", 0.0, 1.0, 0.55, 0.05) if (show_images and show_alveoli_colored) else 0.55
    alveoli_remove_border = st.sidebar.checkbox("Alveoli: remove border-touching", value=True) if (show_images and show_alveoli_colored) else True
    alveoli_color_seed = st.sidebar.number_input("Alveoli color seed", value=0, step=1) if (show_images and show_alveoli_colored) else 0

    # marker strategy controls
    if show_images and show_alveoli_colored:
        st.sidebar.subheader("Alveoli marker strategy")
        marker_method = st.sidebar.selectbox("Marker method", ["local_maxima", "peak_local_max", "h_maxima"], index=0)
        marker_min_distance = st.sidebar.number_input("Marker min distance (px)", value=5, min_value=1, step=1)
        h_maxima_h = st.sidebar.number_input("h-maxima h (px)", value=1.0, min_value=0.0, step=0.1)
        dist_smooth_sigma = st.sidebar.slider("Distance smoothing sigma", 0.0, 5.0, 1.0, 0.1)
        marker_preview = st.sidebar.checkbox("Show marker preview (seeds)", value=True)
    else:
        marker_method = "local_maxima"
        marker_min_distance = 5
        h_maxima_h = 1.0
        dist_smooth_sigma = 1.0
        marker_preview = False

    zip_download = st.sidebar.checkbox("Provide ZIP (tables + optional images)", value=True)

    # -------- Read bytes (no caching) --------
    data = uploaded.getvalue()  # reads file into memory; no disk write
    # Strip EXIF/metadata from the uploaded bytes before decoding (privacy hardening)
    data = remove_exif_bytes(data)
    rgb = read_rgb_from_bytes(data)

    params = {
        "microns_per_pixel": microns_per_pixel,
        "color_target": color_target,
        "hue_min": float(hue_min),
        "hue_max": float(hue_max),
        "sat_thresh": float(sat_thresh),
        "tissue_min_object_size": int(tissue_min_object_size),
        "septa_max_px": float(septa_max_px),
        "inpaint": bool(inpaint),
        "air_min_object_size": int(air_min_object_size),
        "wall_neigh_radius": int(wall_neigh_radius),
        "generate_skeleton": bool(show_images and (show_overlay or show_masks or show_skeleton)),
        "skeleton_prune_px": 0,
        "alveoli_marker_method": str(marker_method),
        "alveoli_marker_min_distance": int(marker_min_distance),
        "alveoli_h_maxima_h": float(h_maxima_h),
        "alveoli_dist_smooth_sigma": float(dist_smooth_sigma),
    }

    # -------- Session-scoped memoization (optional, no persistence) --------
    # This avoids recompute when toggling view components in the SAME session,
    # without using Streamlit's cross-session cache.
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    key = (
        base,
        json.dumps(params, sort_keys=True),
        len(data),
    )

    if key not in st.session_state["cache"]:
        with st.spinner("Segmenting + measuring..."):
            tissue = tissue_mask_advanced(
                rgb,
                min_object_size=int(tissue_min_object_size),
                inpaint=bool(inpaint),
                color_target=str(color_target),
                hue_min=float(hue_min),
                hue_max=float(hue_max),
                sat_thresh=float(sat_thresh),
                septa_max_px=float(septa_max_px),
            )
            df, tissue_mask, skel, thickness_px_map = measure_airspaces_and_wall_thickness(
                tissue,
                microns_per_pixel=microns_per_pixel,
                air_min_object_size=int(air_min_object_size),
                wall_neigh_radius=int(wall_neigh_radius),
                generate_skeleton=bool(params["generate_skeleton"]),
                skeleton_prune_px=0,
            )
        st.session_state["cache"][key] = (df, tissue_mask, skel, thickness_px_map)
    else:
        df, tissue_mask, skel, thickness_px_map = st.session_state["cache"][key]

    # -------- Main outputs --------
    if show_images and show_original:
        st.subheader("Original image")
        st.image(rgb, width='stretch')

    if show_images and show_color_preview and color_target != "none":
        try:
            cm = color_mask_from_rgb(
                rgb,
                color_target=color_target,
                hue_min=hue_min,
                hue_max=hue_max,
                sat_thresh=sat_thresh,
            )
            preview = rgb.copy()
            preview[cm] = [255, 0, 0]
            st.subheader("Hue-target preview (red = selected pixels)")
            st.image(preview, width='stretch')
            st.caption(f"Selected pixels: {int(cm.sum())}")
        except Exception:
            pass

    st.subheader("Key counts")
    st.write(f"Airspaces measured: **{df.shape[0]}**")
    if params["generate_skeleton"]:
        st.write(f"Wall skeleton points: **{int(np.sum(skel))}**")

    # Choose units for summary
    if microns_per_pixel is None:
        summary_cols = ("thickness_px", "diameter_px", "area_px")
        st.caption("Units: thickness/diameter in **pixels** (set microns-per-pixel for µm).")
    else:
        summary_cols = ("thickness_um", "diameter_um", "area_um2")
        st.caption("Units: thickness/diameter in **µm** (pixel columns retained).")

    st.subheader("Summary statistics")
    stats_tbl = compute_stats_table(
        df,
        columns=summary_cols,
        include_ci=bool(include_ci),
        n_boot=int(n_boot),
        ci=float(ci_level),
        seed=int(stats_seed),
    )
    st.dataframe(stats_tbl, width='stretch')

    st.subheader("Correlations (Pearson)")
    corr = corr_table(df, summary_cols)
    if corr.empty:
        st.write("Not enough numeric data.")
    else:
        st.dataframe(corr, width='stretch')

    st.subheader("Raw measurements (first 500 rows)")
    st.dataframe(df.head(500), width='stretch')

    # Downloads (tables)
    st.download_button(
        "Download measurements CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{base}_airspaces.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download stats table CSV",
        stats_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"{base}_stats.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download correlation CSV",
        corr.to_csv(index=True).encode("utf-8") if not corr.empty else b"",
        file_name=f"{base}_correlation.csv",
        mime="text/csv",
    )

    # Thickness map NPY (no identifying metadata)
    npy_buf = BytesIO()
    np.save(npy_buf, thickness_px_map)
    npy_buf.seek(0)
    st.download_button(
        "Download thickness map (NPY)",
        npy_buf.getvalue(),
        file_name=f"{base}_thickness_map.npy",
        mime="application/octet-stream",
    )

    # -------- Optional images --------
    overlay_png = heatmap_png = heat_overlay_png = mask_png = skel_png = alveoli_png = hist_png = None

    if show_images:
        if show_hists and df.shape[0] > 0:
            st.subheader("Thickness distribution (hist + box)")
            col = summary_cols[0]
            hist_png = plot_histogram_and_box_bytes(df, col, bins=60)
            if hist_png:
                st.image(hist_png, width='stretch')
                st.download_button(
                    "Download histogram PNG",
                    hist_png,
                    file_name=f"{base}_hist_{col}.png",
                    mime="image/png",
                )

        if show_overlay:
            sk = skel if show_skeleton else None
            overlay_img = make_edge_overlay(rgb, tissue_mask, sk)
            st.subheader("Edge overlay (edges green; skeleton red if enabled)")
            st.image(overlay_img, width='stretch')
            _, buf = cv2.imencode(".png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            overlay_png = buf.tobytes()
            st.download_button(
                "Download overlay PNG",
                overlay_png,
                file_name=f"{base}_overlay.png",
                mime="image/png",
            )

        if show_heatmap:
            heat_rgb = thickness_map_to_colormap_image(thickness_px_map, cmap="viridis")
            heat_overlay = overlay_heatmap_on_rgb(rgb, heat_rgb, alpha=float(heat_alpha))

            st.subheader("Thickness heatmap overlay")
            st.image(heat_overlay, width='stretch')

            _, bufh = cv2.imencode(".png", cv2.cvtColor(heat_overlay, cv2.COLOR_RGB2BGR))
            heat_overlay_png = bufh.tobytes()
            st.download_button(
                "Download heatmap overlay PNG",
                heat_overlay_png,
                file_name=f"{base}_heatmap_overlay.png",
                mime="image/png",
            )

            _, bufc = cv2.imencode(".png", cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
            heatmap_png = bufc.tobytes()
            st.download_button(
                "Download thickness colormap PNG",
                heatmap_png,
                file_name=f"{base}_thickness_colormap.png",
                mime="image/png",
            )

        if show_masks:
            st.subheader("Masks")
            mask_img = (tissue_mask.astype(np.uint8) * 255)
            st.write("Tissue mask")
            st.image(mask_img, width='stretch')
            _, bufm = cv2.imencode(".png", mask_img)
            mask_png = bufm.tobytes()
            st.download_button(
                "Download mask PNG",
                mask_png,
                file_name=f"{base}_mask.png",
                mime="image/png",
            )

            if params["generate_skeleton"]:
                skel_img = (skel.astype(np.uint8) * 255)
                st.write("Skeleton")
                st.image(skel_img, width='stretch')
                _, bufs = cv2.imencode(".png", skel_img)
                skel_png = bufs.tobytes()
                st.download_button(
                    "Download skeleton PNG",
                    skel_png,
                    file_name=f"{base}_skeleton.png",
                    mime="image/png",
                )

        if show_alveoli_colored:
            if marker_preview:
                try:
                    air_for_markers = (~tissue_mask.astype(bool)).astype(bool)
                    try:
                        air_for_markers = morphology.remove_small_objects(
                            air_for_markers,
                            min_size=int(max(1, air_min_object_size)),
                        )
                    except Exception:
                        pass
                    markers_img, _ = compute_alveoli_markers(
                        air_for_markers,
                        air_min_object_size=int(air_min_object_size),
                        marker_method=str(marker_method),
                        marker_min_distance=int(marker_min_distance),
                        h_maxima_h=float(h_maxima_h),
                        dist_smooth_sigma=float(dist_smooth_sigma),
                    )
                    markers_mask = markers_img > 0
                    try:
                        markers_vis = morphology.binary_dilation(markers_mask, footprint=morphology.disk(1))
                    except Exception:
                        markers_vis = markers_mask
                    preview_img = rgb.copy()
                    preview_img[markers_vis] = [255, 0, 0]
                    st.subheader("Marker preview (red = watershed seeds)")
                    st.image(preview_img, width='stretch')
                    st.caption(f"Markers detected: {int(np.max(markers_img))}")
                except Exception:
                    pass

            colored, n_alv = alveoli_colored_overlay(
                rgb,
                tissue_mask,
                air_min_object_size=int(air_min_object_size),
                remove_border=bool(alveoli_remove_border),
                alpha=float(alveoli_overlay_alpha),
                seed=int(alveoli_color_seed),
                marker_method=str(marker_method),
                marker_min_distance=int(marker_min_distance),
                h_maxima_h=float(h_maxima_h),
                dist_smooth_sigma=float(dist_smooth_sigma),
            )
            st.subheader("Coloured alveoli overlay")
            st.write(f"Identified alveoli: **{n_alv}**")
            st.image(colored, width='stretch')
            _, buf = cv2.imencode(".png", cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
            alveoli_png = buf.tobytes()
            st.download_button(
                "Download coloured alveoli overlay PNG",
                alveoli_png,
                file_name=f"{base}_alveoli_coloured_overlay.png",
                mime="image/png",
            )

    # -------- ZIP bundle --------
    if zip_download:
        mem = BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{base}_airspaces.csv", df.to_csv(index=False))
            zf.writestr(f"{base}_stats.csv", stats_tbl.to_csv(index=False))
            zf.writestr(f"{base}_correlation.csv", corr.to_csv(index=True) if not corr.empty else "")

            # Privacy-minimal metadata: no filenames/paths, no uploaded_name.
            zf.writestr(
                f"{base}_metadata.json",
                json.dumps(
                    {
                        "anonymized": bool(anonymize_outputs),
                        "image_shape": list(rgb.shape),
                        "params": params,
                        "n_airspaces": int(df.shape[0]),
                        "stats": {"include_ci": include_ci, "n_boot": int(n_boot), "ci": float(ci_level), "seed": int(stats_seed)},
                    },
                    indent=2,
                ),
            )

            tbuf = BytesIO()
            np.save(tbuf, thickness_px_map)
            tbuf.seek(0)
            zf.writestr(f"{base}_thickness_map.npy", tbuf.read())

            if overlay_png is not None:
                zf.writestr(f"{base}_overlay.png", overlay_png)
            if heatmap_png is not None:
                zf.writestr(f"{base}_thickness_colormap.png", heatmap_png)
            if heat_overlay_png is not None:
                zf.writestr(f"{base}_heatmap_overlay.png", heat_overlay_png)
            if mask_png is not None:
                zf.writestr(f"{base}_mask.png", mask_png)
            if skel_png is not None:
                zf.writestr(f"{base}_skeleton.png", skel_png)
            if alveoli_png is not None:
                zf.writestr(f"{base}_alveoli_coloured_overlay.png", alveoli_png)
            if hist_png is not None:
                zf.writestr(f"{base}_hist_{summary_cols[0]}.png", hist_png)

        mem.seek(0)
        st.download_button(
            "Download ZIP",
            mem.getvalue(),
            file_name=f"{base}_bundle.zip",
            mime="application/zip",
        )

    # best-effort cleanup (still only in-memory)
    try:
        del data
    except Exception:
        pass


# =========================
# CLI (privacy maximised)
# =========================

def detect_microns_from_filename(name: str) -> Optional[float]:
    patterns = [
        r"mpp([0-9]*\.[0-9]+)",
        r"([0-9]*\.[0-9]+)um",
        r"_([0-9]*\.[0-9]+)",
        r"_([0-9]+)p([0-9]+)",
    ]
    for pat in patterns:
        m = re.findall(pat, name)
        if not m:
            continue
        val = m[-1]
        try:
            if isinstance(val, tuple):
                val = ".".join(val)
            return float(val)
        except Exception:
            pass
    return None


@dataclass
class CliArgs:
    image: Path
    out_dir: Path
    microns_per_pixel: Optional[float]
    auto_mpp: bool
    verbose: bool

    # privacy
    keep_names: bool  # if False, anonymize everything (default)

    # segmentation
    color_target: str
    hue_min: float
    hue_max: float
    sat_thresh: float
    tissue_min_object_size: int
    septa_max_px: float
    inpaint: bool

    # measurement
    air_min_object_size: int
    wall_neigh_radius: int

    # stats
    include_ci: bool
    n_boot: int
    ci_level: float
    seed: int

    # images optional
    save_images: bool
    overlay: bool
    skeleton: bool
    masks: bool
    heatmap: bool
    alveoli_overlay: bool
    alveoli_remove_border: bool
    alveoli_alpha: float
    alveoli_seed: int

    # marker options
    marker_method: str
    marker_min_distance: int
    h_maxima_h: float
    dist_smooth_sigma: float


def parse_args() -> CliArgs:
    ap = argparse.ArgumentParser(description="Alveolar airspaces + wall thickness (privacy-maximised)")
    ap.add_argument("image", help="Path to image (png/jpg/tif).")
    ap.add_argument("--out_dir", default="out_airspaces", help="Output folder.")
    ap.add_argument("--microns_per_pixel", type=float, default=None)
    ap.add_argument("--auto_mpp", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # Privacy: anonymize by default; --keep_names disables anonymization.
    ap.add_argument("--keep_names", action="store_true", help="Keep original filename stems/paths in outputs (NOT recommended).")

    ap.add_argument("--color_target", default="purple", choices=["purple", "pink", "none", "custom"])
    ap.add_argument("--hue_min", type=float, default=0.65)
    ap.add_argument("--hue_max", type=float, default=0.90)
    ap.add_argument("--sat_thresh", type=float, default=0.05)
    ap.add_argument("--tissue_min_object_size", type=int, default=150)
    ap.add_argument("--septa_max_px", type=float, default=8.0)
    ap.add_argument("--inpaint", action="store_true")

    ap.add_argument("--air_min_object_size", type=int, default=16)
    ap.add_argument("--wall_neigh_radius", type=int, default=3)

    ap.add_argument("--include_ci", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--save_images", action="store_true", help="Enable saving images (off by default).")
    ap.add_argument("--overlay", action="store_true", help="Save edge overlay (requires --save_images).")
    ap.add_argument("--skeleton", action="store_true", help="Overlay skeleton in red (requires --overlay).")
    ap.add_argument("--masks", action="store_true", help="Save mask PNG (requires --save_images).")
    ap.add_argument("--heatmap", action="store_true", help="Save heatmap overlay + colormap (requires --save_images).")
    ap.add_argument("--alveoli_overlay", action="store_true", help="Save coloured alveoli overlay (requires --save_images).")
    ap.add_argument("--alveoli_remove_border", action="store_true")
    ap.add_argument("--alveoli_alpha", type=float, default=0.55)
    ap.add_argument("--alveoli_seed", type=int, default=0)

    ap.add_argument("--marker_method", type=str, default="local_maxima", choices=["local_maxima", "peak_local_max", "h_maxima"])
    ap.add_argument("--marker_min_distance", type=int, default=5)
    ap.add_argument("--h_maxima_h", type=float, default=1.0)
    ap.add_argument("--dist_smooth_sigma", type=float, default=1.0)

    a = ap.parse_args()
    return CliArgs(
        image=Path(a.image),
        out_dir=Path(a.out_dir),
        microns_per_pixel=a.microns_per_pixel,
        auto_mpp=bool(a.auto_mpp),
        verbose=bool(a.verbose),
        keep_names=bool(a.keep_names),

        color_target=str(a.color_target),
        hue_min=float(a.hue_min),
        hue_max=float(a.hue_max),
        sat_thresh=float(a.sat_thresh),
        tissue_min_object_size=int(a.tissue_min_object_size),
        septa_max_px=float(a.septa_max_px),
        inpaint=bool(a.inpaint),

        air_min_object_size=int(a.air_min_object_size),
        wall_neigh_radius=int(a.wall_neigh_radius),

        include_ci=bool(a.include_ci),
        n_boot=int(a.n_boot),
        ci_level=float(a.ci_level),
        seed=int(a.seed),

        save_images=bool(a.save_images),
        overlay=bool(a.overlay),
        skeleton=bool(a.skeleton),
        masks=bool(a.masks),
        heatmap=bool(a.heatmap),
        alveoli_overlay=bool(a.alveoli_overlay),
        alveoli_remove_border=bool(a.alveoli_remove_border),
        alveoli_alpha=float(a.alveoli_alpha),
        alveoli_seed=int(a.alveoli_seed),

        marker_method=str(a.marker_method),
        marker_min_distance=int(a.marker_min_distance),
        h_maxima_h=float(a.h_maxima_h),
        dist_smooth_sigma=float(a.dist_smooth_sigma),
    )


def cli_main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not args.image.exists():
        logging.error("Image not found.")
        sys.exit(1)

    if args.auto_mpp and args.microns_per_pixel is None:
        det = detect_microns_from_filename(args.image.stem)
        if det is not None:
            args.microns_per_pixel = det
            logging.info("Auto-detected microns_per_pixel=%s from filename.", det)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rgb = read_rgb_image(args.image)

    anonymize = not args.keep_names
    base_name = uuid.uuid4().hex if anonymize else safe_stem(args.image.stem)

    tissue = tissue_mask_advanced(
        rgb,
        min_object_size=args.tissue_min_object_size,
        inpaint=args.inpaint,
        color_target=args.color_target,
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_thresh=args.sat_thresh,
        septa_max_px=args.septa_max_px,
    )

    need_skel = bool(args.save_images and (args.overlay and args.skeleton))
    df, tissue_mask, skel, thickness_px_map = measure_airspaces_and_wall_thickness(
        tissue,
        microns_per_pixel=args.microns_per_pixel,
        air_min_object_size=args.air_min_object_size,
        wall_neigh_radius=args.wall_neigh_radius,
        generate_skeleton=need_skel,
        skeleton_prune_px=0,
    )

    if args.microns_per_pixel is None:
        summary_cols = ("thickness_px", "diameter_px", "area_px")
    else:
        summary_cols = ("thickness_um", "diameter_um", "area_um2")

    stats_tbl = compute_stats_table(
        df,
        columns=summary_cols,
        include_ci=args.include_ci,
        n_boot=args.n_boot,
        ci=args.ci_level,
        seed=args.seed,
    )
    corr = corr_table(df, summary_cols)

    # Save tables (anonymized stem by default)
    points_csv = args.out_dir / f"{base_name}_airspaces.csv"
    stats_csv = args.out_dir / f"{base_name}_stats.csv"
    corr_csv = args.out_dir / f"{base_name}_correlation.csv"
    df.to_csv(points_csv, index=False)
    stats_tbl.to_csv(stats_csv, index=False)
    corr.to_csv(corr_csv, index=True)

    # Thickness map always saved (anonymized stem by default)
    np.save(str(args.out_dir / f"{base_name}_thickness_map.npy"), thickness_px_map)

    # Optional images
    if args.save_images:
        if args.overlay:
            overlay_img = make_edge_overlay(rgb, tissue_mask, skel if args.skeleton else None)
            save_rgb_png(overlay_img, args.out_dir / f"{base_name}_overlay.png")

        if args.masks:
            save_binary_png(tissue_mask, args.out_dir / f"{base_name}_mask.png")
            if need_skel:
                save_binary_png(skel, args.out_dir / f"{base_name}_skeleton.png")

        if args.heatmap:
            heat = thickness_map_to_colormap_image(thickness_px_map, cmap="viridis")
            save_rgb_png(heat, args.out_dir / f"{base_name}_thickness_colormap.png")
            heat_overlay = overlay_heatmap_on_rgb(rgb, heat, alpha=0.6)
            save_rgb_png(heat_overlay, args.out_dir / f"{base_name}_heatmap_overlay.png")

        if args.alveoli_overlay:
            colored, _ = alveoli_colored_overlay(
                rgb,
                tissue_mask,
                air_min_object_size=args.air_min_object_size,
                remove_border=args.alveoli_remove_border,
                alpha=args.alveoli_alpha,
                seed=args.alveoli_seed,
                marker_method=str(args.marker_method),
                marker_min_distance=int(args.marker_min_distance),
                h_maxima_h=float(args.h_maxima_h),
                dist_smooth_sigma=float(args.dist_smooth_sigma),
            )
            save_rgb_png(colored, args.out_dir / f"{base_name}_alveoli_coloured_overlay.png")

    # Privacy-minimal metadata
    meta = {
        "anonymized": bool(anonymize),
        "image_shape": list(rgb.shape),
        "microns_per_pixel": args.microns_per_pixel,
        "params": {
            "segmentation": {
                "color_target": args.color_target,
                "hue_min": args.hue_min,
                "hue_max": args.hue_max,
                "sat_thresh": args.sat_thresh,
                "tissue_min_object_size": args.tissue_min_object_size,
                "septa_max_px": args.septa_max_px,
                "inpaint": args.inpaint,
            },
            "measurement": {
                "air_min_object_size": args.air_min_object_size,
                "wall_neigh_radius": args.wall_neigh_radius,
            },
            "stats": {
                "include_ci": args.include_ci,
                "n_boot": args.n_boot,
                "ci_level": args.ci_level,
                "seed": args.seed,
                "summary_cols": list(summary_cols),
            },
            "images_enabled": args.save_images,
        },
        "n_airspaces": int(df.shape[0]),
    }
    # Only include path/name if user explicitly requested it.
    if args.keep_names:
        meta["image"] = str(args.image)

    with open(args.out_dir / f"{base_name}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Logs: avoid printing raw path unless keep_names is enabled
    logging.info("Saved outputs under stem: %s", base_name)
    logging.info(" - %s", points_csv)
    logging.info(" - %s", stats_csv)
    logging.info(" - %s", corr_csv)
    logging.info(" - %s", args.out_dir / f"{base_name}_thickness_map.npy")
    logging.info(" - %s", args.out_dir / f"{base_name}_metadata.json")

    if df.shape[0] == 0:
        logging.warning("No airspaces found.")
        sys.exit(2)

    tcol = "thickness_um" if ("thickness_um" in df.columns and df["thickness_um"].notna().any()) else "thickness_px"
    s = _finite_series(df, tcol).to_numpy(dtype=float)
    logging.info("%s: n=%d, median=%0.3f, p05–p95=%0.3f–%0.3f",
                 tcol, s.size, float(np.median(s)), float(np.quantile(s, 0.05)), float(np.quantile(s, 0.95)))


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    # If streamlit is available and no non-flag args are provided -> run app; otherwise run CLI.
    try:
        import streamlit  # noqa: F401

        non_flags = [a for a in sys.argv[1:] if not a.startswith("-")]
        if non_flags:
            cli_main()
        else:
            streamlit_app()
    except Exception:
        cli_main()
