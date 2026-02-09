#!/usr/bin/env python3
"""
DEC + Alveolar Batch Counter — FULL REWRITE (same outputs)

Streamlit:
  streamlit run dec_alv_app.py

Outputs (same as your current app):
- DEC_results.xlsx (sheet: DEC_counts; + DEC_particles if enabled; + errors if any)
- ALV_results.xlsx (sheet: ALV_counts; + ALV_particles if enabled; + errors if any)
- dec_alv_outputs_bundle.zip
  - DEC_results.xlsx, ALV_results.xlsx, params.json
  - errors.csv (if any)
  - qc/{sample_id}_original.jpg, {sample_id}_dec.jpg, {sample_id}_alv_long.jpg, {sample_id}_alv_short.jpg

Notes:
- Image-processing logic is preserved as close as possible to your current app.
- Fiji/ImageJ headless option for ALV produces counts/areas using Analyze Particles and writes Results CSV (no Excel plugin).
"""

import io
import json
import os
import re
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

import cv2
from PIL import Image
from skimage import measure, morphology, exposure


# -----------------------------
# Constants
# -----------------------------
SUPPORTED_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


# -----------------------------
# Privacy + I/O helpers
# -----------------------------
def safe_stem(name: str) -> str:
    name = name or "image"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def remove_exif_bytes(b: bytes) -> bytes:
    """Best-effort EXIF/metadata removal by re-saving as PNG in memory."""
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return b


def read_image_bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")
    return bgr


def bgr_to_jpeg_bytes(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def _ensure_gray_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return g.astype(np.uint8)


def bgr_to_gray8(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _ensure_gray_uint8(gray)


def load_zip_images(zb: bytes, strip_exif: bool) -> Dict[str, bytes]:
    images: Dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext in SUPPORTED_EXTS:
                raw = zf.read(info.filename)
                images[Path(info.filename).name] = remove_exif_bytes(raw) if strip_exif else raw
    return images


def load_multi_uploaded_images(uploaded_files, strip_exif: bool) -> Dict[str, bytes]:
    """
    Return {basename: bytes} from:
      - multiple images
      - one image
      - one or more ZIPs
    If a ZIP and image(s) contain the same basename, later items win.
    """
    if not uploaded_files:
        return {}

    images: Dict[str, bytes] = {}

    for uf in uploaded_files:
        if uf is None:
            continue

        name = getattr(uf, "name", "")
        ext = Path(name).suffix.lower()

        if ext == ".zip":
            images.update(load_zip_images(uf.getvalue(), strip_exif=strip_exif))
            continue

        if ext in SUPPORTED_EXTS:
            raw = uf.getvalue()
            images[Path(name).name] = remove_exif_bytes(raw) if strip_exif else raw
            continue

        raise ValueError(
            f"Unsupported upload type: {name}. Please upload .zip or one/more images ({', '.join(sorted(SUPPORTED_EXTS))})."
        )

    return images


# -----------------------------
# Unit conversion helpers
# -----------------------------
def pixel_area_mm2_from_um_per_px(um_per_px: float) -> float:
    """Convert linear calibration (µm/px) into area conversion (mm²/px²)."""
    if um_per_px is None or not np.isfinite(um_per_px) or um_per_px <= 0:
        return float("nan")
    # (um/px)^2 => um^2/px^2 ; 1 um^2 = 1e-6 mm^2
    return float((um_per_px ** 2) * 1e-6)


def normalize_area_fraction(sum_area_px: float, image_h: int, image_w: int) -> float:
    """Return sum_area_px / (image area in px)."""
    if image_h is None or image_w is None:
        return float("nan")
    if image_h <= 0 or image_w <= 0:
        return float("nan")
    denom = float(image_h * image_w)
    if denom <= 0:
        return float("nan")
    return float(sum_area_px) / denom


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()


# -----------------------------
# Bundle ZIP builder (same structure)
# -----------------------------
def make_bundle_zip(
    dec_excel: bytes,
    alv_excel: bytes,
    qc_items: List[Tuple[str, bytes, bytes, bytes, bytes]],  # (sample_id, orig, dec, long, short)
    params: dict,
    errors_df: pd.DataFrame,
    name_map_df: Optional[pd.DataFrame],
) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("DEC_results.xlsx", dec_excel)
        z.writestr("ALV_results.xlsx", alv_excel)
        z.writestr("params.json", json.dumps(params, indent=2))

        if errors_df is not None and not errors_df.empty:
            z.writestr("errors.csv", errors_df.to_csv(index=False))

        if name_map_df is not None and not name_map_df.empty:
            z.writestr("name_map.csv", name_map_df.to_csv(index=False))

        for sid, orig_j, dec_j, long_j, short_j in qc_items:
            if orig_j:
                z.writestr(f"qc/{sid}_original.jpg", orig_j)
            if dec_j:
                z.writestr(f"qc/{sid}_dec.jpg", dec_j)
            if long_j:
                z.writestr(f"qc/{sid}_alv_long.jpg", long_j)
            if short_j:
                z.writestr(f"qc/{sid}_alv_short.jpg", short_j)

        z.writestr(
            "README.txt",
            "Bundle contents:\n"
            "- DEC_results.xlsx\n"
            "- ALV_results.xlsx\n"
            "- params.json\n"
            "- errors.csv (if any)\n"
            "- name_map.csv (if enabled)\n"
            "- qc/ overlays (subset)\n"
        )

    bio.seek(0)
    return bio.getvalue()


# -----------------------------
# Image processing primitives (same family)
# -----------------------------
def rolling_ball_background_subtract(gray8: np.ndarray, radius: int = 50) -> np.ndarray:
    k = max(3, int(radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray8, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(gray8, background)


def unsharp_mask(gray8: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(gray8, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def find_edges(gray8: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def threshold_mask_like_inter(mag8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(mag8, (0, 0), 1.0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (m > 0).astype(np.uint8)


def enhance_contrast_like(gray8: np.ndarray, sat_pct: float) -> np.ndarray:
    lo = np.percentile(gray8, sat_pct / 2.0)
    hi = np.percentile(gray8, 100 - sat_pct / 2.0)
    if hi <= lo:
        return gray8
    return exposure.rescale_intensity(gray8, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def make_binary(gray8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray8, (0, 0), 1.0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (m > 0).astype(np.uint8)


def close_morph(mask01: np.ndarray, k: int = 3) -> np.ndarray:
    k = max(3, int(k) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(mask01.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return (out > 0).astype(np.uint8)


def mask_to_overlay(bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = bgr.copy().astype(np.float32)
    green = np.zeros_like(overlay)
    green[..., 1] = 255.0
    m = mask01.astype(bool)
    overlay[m] = (1 - alpha) * overlay[m] + alpha * green[m]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_contours_from_mask(bgr: np.ndarray, mask01: np.ndarray, color=(0, 255, 255), thickness=1) -> np.ndarray:
    out = bgr.copy()
    contours, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def safe_mode_intensity(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    vals = values.astype(np.uint8)
    hist = np.bincount(vals, minlength=256)
    return float(np.argmax(hist))


# -----------------------------
# DEC pipeline (same outputs)
# -----------------------------
@dataclass(frozen=True)
class DECParams:
    area_min: float = 7
    area_max: float = 30
    circ_min: float = 0.85
    circ_max: float = 1.00
    mode_cutoff: float = 120
    round_min: float = 0.5
    ar_max: float = 1.4
    min_object_pixels: int = 5


def region_props_from_binary(mask01: np.ndarray, gray_original: np.ndarray) -> pd.DataFrame:
    labeled = measure.label(mask01, connectivity=2)
    props = measure.regionprops(labeled, intensity_image=gray_original)

    rows = []
    for p in props:
        area = float(p.area)
        mean_int = float(p.mean_intensity) if p.mean_intensity is not None else np.nan

        coords = p.coords
        pix = gray_original[coords[:, 0], coords[:, 1]]
        mode_int = safe_mode_intensity(pix)

        perim = float(p.perimeter) if p.perimeter and p.perimeter > 0 else np.nan
        circularity = (4 * np.pi * area / (perim ** 2)) if (perim and perim > 0) else np.nan

        maj = float(p.major_axis_length) if p.major_axis_length else np.nan
        mino = float(p.minor_axis_length) if p.minor_axis_length else np.nan
        ar = (maj / mino) if (maj and mino and mino > 0) else np.nan
        roundness = (4 * area / (np.pi * (maj ** 2))) if (maj and maj > 0) else np.nan

        rows.append(
            {
                "label": int(p.label),
                "area": area,
                "mean": mean_int,
                "mode": mode_int,
                "circularity": circularity,
                "AR": ar,
                "round": roundness,
            }
        )

    return pd.DataFrame(rows)


def apply_particle_filters(
    df: pd.DataFrame,
    area_min: float,
    area_max: float,
    circ_min: float,
    circ_max: float,
    mode_cutoff: float,
    round_min: float,
    ar_max: float,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out[(out["area"] >= area_min) & (out["area"] <= area_max)]
    out = out[(out["circularity"].isna()) | ((out["circularity"] >= circ_min) & (out["circularity"] <= circ_max))]
    out = out[(out["mode"].isna()) | (out["mode"] <= mode_cutoff)]
    out = out[(out["round"].isna()) | (out["round"] >= round_min)]
    out = out[(out["AR"].isna()) | (out["AR"] <= ar_max)]
    return out.reset_index(drop=True)


def run_dec_pipeline(bgr: np.ndarray, params: DECParams) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    gray = bgr_to_gray8(bgr)
    original = gray.copy()

    edges = find_edges(gray)
    mask01 = threshold_mask_like_inter(edges)

    mask01 = morphology.remove_small_objects(mask01.astype(bool), min_size=params.min_object_pixels).astype(np.uint8)

    particles_all = region_props_from_binary(mask01, original)
    particles_kept = apply_particle_filters(
        particles_all,
        area_min=params.area_min,
        area_max=params.area_max,
        circ_min=params.circ_min,
        circ_max=params.circ_max,
        mode_cutoff=params.mode_cutoff,
        round_min=params.round_min,
        ar_max=params.ar_max,
    )

    kept_mask = np.zeros_like(mask01, dtype=np.uint8)
    if not particles_kept.empty:
        labeled = measure.label(mask01, connectivity=2)
        keep_labels = particles_kept["label"].astype(int).to_numpy()
        kept_mask = np.isin(labeled, keep_labels).astype(np.uint8)

    overlay = mask_to_overlay(bgr, kept_mask, alpha=0.45)
    overlay = draw_contours_from_mask(overlay, kept_mask)

    return particles_all, particles_kept, mask01, overlay


# -----------------------------
# ALV pipeline (Python)
# -----------------------------
@dataclass(frozen=True)
class ALVParams:
    rolling: int = 50
    sharpen_amount: float = 1.0
    contrast_sat_pct: float = 20.0
    blur_sigma: float = 2.0
    # Macro: Analyze Particles size=100-Infinity pixel
    min_area: float = 100


def analyze_particles_binary(mask01: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    labeled = measure.label(mask01, connectivity=2)
    props = measure.regionprops(labeled)
    rows = [{"label": int(p.label), "area": float(p.area)} for p in props]
    return pd.DataFrame(rows), labeled


def run_alv_pipeline(
    bgr: np.ndarray, params: ALVParams, variant: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    gray = bgr_to_gray8(bgr)

    proc = rolling_ball_background_subtract(gray, radius=params.rolling)
    proc = unsharp_mask(proc, amount=params.sharpen_amount, sigma=1.0)
    proc = enhance_contrast_like(proc, sat_pct=params.contrast_sat_pct)
    proc = find_edges(proc)
    proc = cv2.GaussianBlur(proc, (0, 0), float(params.blur_sigma))

    mask01 = make_binary(proc)
    mask01 = close_morph(mask01, k=3)

    if (variant or "").lower() == "long":
        # mimic "Invert" pathway
        mask01 = (1 - mask01).astype(np.uint8)
        mask01 = close_morph(mask01, k=3)

    particles_all, labeled = analyze_particles_binary(mask01)
    particles_kept = particles_all[particles_all["area"] >= params.min_area].reset_index(drop=True)

    kept_mask = np.zeros_like(mask01, dtype=np.uint8)
    if not particles_kept.empty:
        keep_labels = particles_kept["label"].astype(int).to_numpy()
        kept_mask = np.isin(labeled, keep_labels).astype(np.uint8)

    overlay = mask_to_overlay(bgr, kept_mask, alpha=0.35)
    overlay = draw_contours_from_mask(overlay, kept_mask)

    return particles_all, particles_kept, mask01, overlay


def summarize_alv(particles_kept: pd.DataFrame) -> Tuple[float, float]:
    """Return (mean_area_px, sum_area_px) from kept alveolar particles."""
    if particles_kept is None or particles_kept.empty or "area" not in particles_kept.columns:
        return float("nan"), float("nan")

    areas = pd.to_numeric(particles_kept["area"], errors="coerce").dropna()
    if areas.empty:
        return float("nan"), float("nan")

    return float(areas.mean()), float(areas.sum())


def suggest_variant(n_long: int, n_short: int) -> str:
    # Keep original heuristic, with safe guard for complete failure
    if n_long == 0 and n_short > 0:
        return "short"
    if n_short == 0 and n_long > 0:
        return "long"
    if n_long < 20 and n_short >= 20:
        return "short"
    if n_short < 20 and n_long >= 20:
        return "long"
    if abs(n_short - n_long) <= 2:
        return "either"
    return "review"


# -----------------------------
# Fiji/ImageJ integration (optional)
# -----------------------------
FIJI_DEFAULTS = [
    # macOS
    "/Applications/Fiji.app/ImageJ-macosx",
    "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
    # Linux (common install locations)
    "/opt/Fiji.app/ImageJ-linux64",
    "/usr/local/Fiji.app/ImageJ-linux64",
    # Windows (common install locations)
    "C:/Fiji.app/ImageJ-win64.exe",
    "C:/Program Files/Fiji.app/ImageJ-win64.exe",
    "C:/Program Files (x86)/Fiji.app/ImageJ-win64.exe",
]


def _is_executable_file(p: Path) -> bool:
    if not p.exists() or not p.is_file():
        return False
    # On Windows, the executable bit isn't meaningful.
    if os.name == "nt":
        return p.suffix.lower() in {".exe", ".bat", ".cmd"}
    return os.access(str(p), os.X_OK)


def _find_fiji_executable(user_path: str) -> Optional[str]:
    """Return an executable path to Fiji/ImageJ or None."""
    if user_path:
        p = Path(user_path).expanduser()
        if _is_executable_file(p):
            return str(p)

    for guess in FIJI_DEFAULTS:
        p = Path(guess)
        if _is_executable_file(p):
            return str(p)

    return None


def run_fiji_alv_macro(
    fiji_exe: str,
    img_bytes: bytes,
    variant: str,
    rolling: int,
    saturated: float,
    blur_sigma: float,
    min_area: float,
    scale_distance: int = 1104,
) -> Tuple[int, float, float]:
    """
    Run Fiji headless using a generated macro (long/short), parse Results CSV.

    Returns: (count, mean_area_px, sum_area_px)

    Notes:
    - Uses Analyze Particles with size=min_area-Infinity pixel and exclude edges.
    - Saves Results as CSV (no Excel plugin required).
    """
    variant_l = (variant or "").strip().lower()
    if variant_l not in {"long", "short"}:
        raise ValueError("variant must be 'long' or 'short'")

    macro_template = r"""
    // Expect args like: img=/path/to.png,out=/path/to.csv,rolling=50,sat=20,sigma=2,min_area=100,variant=long,scale_dist=1104
    function getArg(args, key) {
        // returns value string or ""
        items = split(args, ",");
        for (i=0; i<items.length; i++) {
            kv = split(items[i], "=");
            if (kv.length >= 2) {
                k = trim(kv[0]);
                // re-join value in case it contains '='
                v = kv[1];
                if (kv.length > 2) {
                    for (j=2; j<kv.length; j++) v = v + "=" + kv[j];
                }
                if (k == key) return trim(v);
            }
        }
        return "";
    }

    args = getArgument();
    img = getArg(args, "img");
    out = getArg(args, "out");
    rolling = parseInt(getArg(args, "rolling"));
    sat = parseFloat(getArg(args, "sat"));
    sigma = parseFloat(getArg(args, "sigma"));
    min_area = parseFloat(getArg(args, "min_area"));
    variant = getArg(args, "variant");
    scale_dist = parseInt(getArg(args, "scale_dist"));

    open(img);

    // Set Scale (your existing choice)
    run("Set Scale...", "distance="+scale_dist+" known=1 pixel=1 unit=mm global");

    run("Subtract Background...", "rolling="+rolling+" light");
    run("Sharpen");
    run("Enhance Contrast...", "saturated="+sat);
    run("Find Edges");
    run("Gaussian Blur...", "sigma="+sigma);
    setOption("BlackBackground", false);
    run("Make Binary");

    if (toLowerCase(variant) == "long") {
        run("Close-");
        run("Invert");
    }

    run("Set Measurements...", "area perimeter display redirect=None decimal=3");
    run("Analyze Particles...", "size="+min_area+"-Infinity pixel exclude clear");

    // Save Results table as CSV
    saveAs("Results", out);
    run("Close All");
    """


    def ij_arg_escape(p: str) -> str:
        # Fiji args are a single string; normalize separators
        return str(p).replace("\\", "/")

    with tempfile.TemporaryDirectory(prefix="alv_fiji_") as td:
        td_path = Path(td)
        img_path = td_path / "input.png"
        macro_path = td_path / "alv_headless.ijm"
        results_csv = td_path / f"Results_{variant_l}.csv"

        # Always convert to PNG in-memory to mimic 'Stack to RGB' / format normalization.
        img_path.write_bytes(remove_exif_bytes(img_bytes))
        macro_path.write_text(macro_template, encoding="utf-8")

        arg_str = (
            f"img={ij_arg_escape(img_path)},"
            f"out={ij_arg_escape(results_csv)},"
            f"rolling={int(rolling)},"
            f"sat={float(saturated)},"
            f"sigma={float(blur_sigma)},"
            f"min_area={float(min_area)},"
            f"variant={variant_l},"
            f"scale_dist={int(scale_distance)}"
        )

        cmd = [fiji_exe, "--headless", "--run", str(macro_path), arg_str]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Fiji failed (variant={variant_l}).\n"
                f"stdout:\n{proc.stdout[-4000:]}\n\nstderr:\n{proc.stderr[-4000:]}"
            )

        if not results_csv.exists():
            raise RuntimeError(
                f"Fiji did not produce Results CSV (variant={variant_l}).\n"
                f"stdout:\n{proc.stdout[-4000:]}\n\nstderr:\n{proc.stderr[-4000:]}"
            )

        df = pd.read_csv(results_csv)
        if df.empty or "Area" not in df.columns:
            return 0, float("nan"), 0.0

        areas = pd.to_numeric(df["Area"], errors="coerce").dropna()
        if areas.empty:
            return 0, float("nan"), 0.0

        return int(areas.shape[0]), float(areas.mean()), float(areas.sum())


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="DEC + Alveolar Batch Counter", layout="wide")
st.title("DEC + Alveolar Batch Counter")

with st.sidebar:
    st.header("1) Upload")
    upload = st.file_uploader(
        "ZIP of images OR one/more images",
        type=["zip", "tif", "tiff", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    st.header("2) Privacy")
    strip_exif = st.checkbox("Strip EXIF/metadata from uploads (recommended)", value=True)

    st.header("3) Output options")
    include_particle_tables = st.checkbox("Include particle tables (bigger, slower)", value=False)
    qc_preview_n = st.slider("QC preview count", 0, 30, 12, 1)
    qc_jpeg_quality = st.slider("QC JPEG quality", 50, 95, 85, 5)

    st.header("3b) ALV backend")
    alv_backend = st.selectbox(
        "ALV analysis backend",
        options=["Python (built-in)", "Fiji/ImageJ (headless)"],
        index=0,
        help=(
            "Python mode is the default and produces overlays + counts/areas in one pipeline. "
            "Fiji/ImageJ mode runs headless Analyze Particles for counts/areas; overlays still come from Python."
        ),
    )

    use_fiji_for_alv = alv_backend.startswith("Fiji")

    fiji_exe_upload = st.file_uploader(
        "Select Fiji executable (optional)",
        type=["exe", "bat", "cmd"],
        accept_multiple_files=False,
        disabled=not use_fiji_for_alv,
        help=(
            "Windows: select ImageJ-win64.exe (or a .bat/.cmd launcher). "
            "If you don't upload, the app will try default install locations or the manual path below."
        ),
    )
    fiji_path = st.text_input(
        "Or enter Fiji executable path",
        value="",
        disabled=not use_fiji_for_alv,
        help=(
            "Examples: "
            "macOS: /Applications/Fiji.app/ImageJ-macosx | "
            "Windows: C:/Fiji.app/ImageJ-win64.exe"
        ),
    )

    st.header("4) DEC parameters")
    mode_cutoff = st.number_input("Mode cutoff (delete if Mode > cutoff)", 0.0, 255.0, 120.0, 1.0)
    area_min = st.number_input("Particle area min", 1.0, 10000.0, 7.0, 1.0)
    area_max = st.number_input("Particle area max", 1.0, 10000.0, 30.0, 1.0)
    circ_min = st.number_input("Circularity min", 0.0, 1.0, 0.85, 0.01)
    circ_max = st.number_input("Circularity max", 0.0, 1.0, 1.00, 0.01)
    round_min = st.number_input("Round min (delete if Round < min)", 0.0, 2.0, 0.5, 0.05)
    ar_max = st.number_input("AR max (delete if AR > max)", 0.5, 10.0, 1.4, 0.05)

    st.header("5) Alveolar parameters")
    alv_min_area = st.number_input("Alveolar min particle area", 1.0, 1e7, 100.0, 5.0)
    rolling = st.number_input("Background subtract radius (rolling)", 1, 500, 50, 5)
    blur_sigma = st.number_input("Gaussian blur sigma", 0.0, 10.0, 2.0, 0.5)
    sat = st.number_input("Contrast saturation % (approx)", 0.0, 50.0, 20.0, 1.0)

    st.header("6) Calibration (optional)")
    st.caption("Use this if you want surface area in mm² (e.g., fixed 20x setup).")
    um_per_px = st.number_input("Microns per pixel (µm/px)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)

    run_btn = st.button("Run batch analysis", type="primary", width="stretch")

if upload is None:
    st.info("Upload one/more images or a ZIP to begin.")
    st.stop()

try:
    image_bytes = load_multi_uploaded_images(upload, strip_exif=strip_exif)
except Exception as e:
    st.error(str(e))
    st.stop()

if not image_bytes:
    st.warning("No supported images found in upload.")
    st.stop()

orig_names = sorted(image_bytes.keys())

# Use original names directly (no anonymisation)
sample_ids = orig_names
name_map_df = None

st.caption(f"Found {len(orig_names)} image(s).")

if not run_btn:
    st.stop()

# Params objects
dec_params = DECParams(
    area_min=float(area_min),
    area_max=float(area_max),
    circ_min=float(circ_min),
    circ_max=float(circ_max),
    mode_cutoff=float(mode_cutoff),
    round_min=float(round_min),
    ar_max=float(ar_max),
)

alv_params = ALVParams(
    rolling=int(rolling),
    contrast_sat_pct=float(sat),
    blur_sigma=float(blur_sigma),
    min_area=float(alv_min_area),
)

run_params = {
    "privacy": {
        "strip_exif": strip_exif,
    },
    "DECParams": asdict(dec_params),
    "ALVParams": asdict(alv_params),
    "include_particle_tables": include_particle_tables,
    "calibration": {"um_per_px": float(um_per_px)},
    "use_fiji_for_alv": bool(use_fiji_for_alv),
}

# Resolve Fiji executable once
fiji_exe = None
if use_fiji_for_alv:
    # Guard: Streamlit Cloud typically runs on Linux. A Windows .exe cannot be executed there.
    up_name = str(getattr(fiji_exe_upload, "name", "")) if fiji_exe_upload is not None else ""
    up_suffix = Path(up_name).suffix.lower() if up_name else ""
    if os.name != "nt" and up_suffix == ".exe":
        st.error(
            "You selected a Windows Fiji executable (.exe), but this app is running on a non-Windows host (likely Linux/Streamlit Cloud).\n\n"
            "This causes: [Errno 8] Exec format error.\n\n"
            "Fix: switch **ALV analysis backend** to **Python (built-in)**, or run this app on Windows with Fiji installed, "
            "or install Fiji for Linux and point to the Linux executable (e.g., ImageJ-linux64)."
        )
        st.stop()

    # Preferred: user provides a path to an installed Fiji.
    fiji_exe = _find_fiji_executable(fiji_path) if fiji_path else None

    # Optional: user uploads a launcher (primarily useful on Windows local runs)
    # NOTE: this runs the uploaded binary on the server hosting Streamlit.
    if fiji_exe is None and fiji_exe_upload is not None:
        try:
            up_name = str(getattr(fiji_exe_upload, "name", ""))
            suffix = Path(up_name).suffix
            fd, tmp_path = tempfile.mkstemp(prefix="fiji_exe_", suffix=suffix)
            os.close(fd)
            Path(tmp_path).write_bytes(fiji_exe_upload.getvalue())
            try:
                os.chmod(tmp_path, 0o755)
            except Exception:
                pass
            if _is_executable_file(Path(tmp_path)):
                fiji_exe = tmp_path
        except Exception as e:
            st.error(f"Could not use uploaded Fiji executable: {e}")
            st.stop()

    # Fallback: probe common install locations
    if fiji_exe is None:
        fiji_exe = _find_fiji_executable("")

    if not fiji_exe:
        st.error(
            "Fiji/ImageJ not found or not executable.\n\n"
            "Fixes:\n"
            "- Ensure Fiji is installed on the machine running this Streamlit app\n"
            "- Set the executable path (macOS: /Applications/Fiji.app/ImageJ-macosx | Windows: C:/Fiji.app/ImageJ-win64.exe)\n"
            "- If running on Streamlit Cloud/Linux, a Windows .exe will not work; use the Python backend instead"
        )
        st.stop()

# Batch accumulators
dec_counts_rows: List[dict] = []
dec_particles_rows: List[pd.DataFrame] = []

alv_counts_rows: List[dict] = []
alv_particles_rows: List[pd.DataFrame] = []

qc_previews: List[Tuple[str, bytes, bytes, bytes, bytes]] = []
errors: List[dict] = []

progress = st.progress(0)
status = st.empty()

px_area_mm2 = pixel_area_mm2_from_um_per_px(float(um_per_px)) if float(um_per_px) > 0 else float("nan")

for i, (orig_name, sid) in enumerate(zip(orig_names, sample_ids)):
    shown_name = orig_name
    status.write(f"Processing {i+1}/{len(orig_names)}: **{shown_name}**")

    try:
        bgr = read_image_bytes_to_bgr(image_bytes[orig_name])
        h, w = int(bgr.shape[0]), int(bgr.shape[1])

        # ---- DEC ----
        dec_all, dec_kept, _, dec_overlay = run_dec_pipeline(bgr, dec_params)
        dec_count = int(len(dec_kept))
        dec_counts_rows.append({"sample_id": sid, "DEC_count": dec_count})

        if include_particle_tables and not dec_all.empty:
            tmp = dec_all.copy()
            tmp.insert(0, "sample_id", sid)
            keep_set = set(dec_kept["label"].astype(int).tolist()) if not dec_kept.empty else set()
            tmp["kept_after_filters"] = tmp["label"].astype(int).isin(keep_set)
            dec_particles_rows.append(tmp)

        # ---- ALV (Long/Short) ----
        # Always compute Python overlays for QC (fast enough, and keeps your bundle structure consistent).
        # If particle tables are enabled and Fiji isn't used, we reuse Python particles tables.
        alv_all_long, alv_kept_long, _, alv_overlay_long = run_alv_pipeline(bgr, alv_params, "long")
        alv_all_short, alv_kept_short, _, alv_overlay_short = run_alv_pipeline(bgr, alv_params, "short")

        if use_fiji_for_alv and fiji_exe:
            n_long, mean_long_px, sa_long_px = run_fiji_alv_macro(
                fiji_exe=fiji_exe,
                img_bytes=image_bytes[orig_name],
                variant="long",
                rolling=int(rolling),
                saturated=float(sat),
                blur_sigma=float(blur_sigma),
                min_area=float(alv_min_area),
                scale_distance=1104,
            )
            n_short, mean_short_px, sa_short_px = run_fiji_alv_macro(
                fiji_exe=fiji_exe,
                img_bytes=image_bytes[orig_name],
                variant="short",
                rolling=int(rolling),
                saturated=float(sat),
                blur_sigma=float(blur_sigma),
                min_area=float(alv_min_area),
                scale_distance=1104,
            )
        else:
            n_long = int(len(alv_kept_long))
            n_short = int(len(alv_kept_short))
            mean_long_px, sa_long_px = summarize_alv(alv_kept_long)
            mean_short_px, sa_short_px = summarize_alv(alv_kept_short)

        suggestion = suggest_variant(int(n_long), int(n_short))

        # Choose "used" like your current behaviour: short only if suggested == short; else long
        if suggestion == "short":
            n_used = int(n_short)
            mean_used_px, sa_used_px = mean_short_px, sa_short_px
        else:
            n_used = int(n_long)
            mean_used_px, sa_used_px = mean_long_px, sa_long_px

        area_frac_used = normalize_area_fraction(sa_used_px, h, w)
        area_frac_long = normalize_area_fraction(sa_long_px, h, w)
        area_frac_short = normalize_area_fraction(sa_short_px, h, w)

        # Physical units if calibration provided
        mean_used_mm2 = _safe_float(mean_used_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")
        sa_used_mm2 = _safe_float(sa_used_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")

        mean_long_mm2 = _safe_float(mean_long_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")
        sa_long_mm2 = _safe_float(sa_long_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")

        mean_short_mm2 = _safe_float(mean_short_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")
        sa_short_mm2 = _safe_float(sa_short_px) * px_area_mm2 if np.isfinite(px_area_mm2) else float("nan")

        alv_counts_rows.append(
            {
                "sample_id": sid,
                "ALV_long_count": int(n_long),
                "ALV_short_count": int(n_short),
                "suggested": suggestion,
                "ALV_n_used": int(n_used),
                # used (px)
                "ALV_mean_size_used_px": _safe_float(mean_used_px),
                "ALV_surface_area_used_px": _safe_float(sa_used_px),
                # used (recommended)
                "ALV_surface_area_used_frac": _safe_float(area_frac_used),
                "ALV_mean_size_used_mm2": _safe_float(mean_used_mm2),
                "ALV_surface_area_used_mm2": _safe_float(sa_used_mm2),
                # long
                "ALV_mean_size_long_px": _safe_float(mean_long_px),
                "ALV_surface_area_long_px": _safe_float(sa_long_px),
                "ALV_surface_area_long_frac": _safe_float(area_frac_long),
                "ALV_mean_size_long_mm2": _safe_float(mean_long_mm2),
                "ALV_surface_area_long_mm2": _safe_float(sa_long_mm2),
                # short
                "ALV_mean_size_short_px": _safe_float(mean_short_px),
                "ALV_surface_area_short_px": _safe_float(sa_short_px),
                "ALV_surface_area_short_frac": _safe_float(area_frac_short),
                "ALV_mean_size_short_mm2": _safe_float(mean_short_mm2),
                "ALV_surface_area_short_mm2": _safe_float(sa_short_mm2),
                # image size
                "image_h_px": h,
                "image_w_px": w,
            }
        )

        # Particle tables: keep schema identical to your current app
        if include_particle_tables:
            # DEC particles already handled above
            # ALV: use Python-derived tables (even if Fiji used) because they exist and match previous output shape.
            if not alv_all_long.empty:
                tl = alv_all_long.copy()
                tl.insert(0, "sample_id", sid)
                tl.insert(1, "variant", "long")
                tl["kept_after_filters"] = tl["area"] >= alv_params.min_area
                alv_particles_rows.append(tl)

            if not alv_all_short.empty:
                ts = alv_all_short.copy()
                ts.insert(0, "sample_id", sid)
                ts.insert(1, "variant", "short")
                ts["kept_after_filters"] = ts["area"] >= alv_params.min_area
                alv_particles_rows.append(ts)

        # QC preview (store JPEG bytes only)
        if len(qc_previews) < int(qc_preview_n):
            orig_j = bgr_to_jpeg_bytes(bgr, quality=int(qc_jpeg_quality))
            dec_j = bgr_to_jpeg_bytes(dec_overlay, quality=int(qc_jpeg_quality))
            long_j = bgr_to_jpeg_bytes(alv_overlay_long, quality=int(qc_jpeg_quality))
            short_j = bgr_to_jpeg_bytes(alv_overlay_short, quality=int(qc_jpeg_quality))
            qc_previews.append((sid, orig_j, dec_j, long_j, short_j))

    except Exception as e:
        errors.append({"sample_id": sid, "error": str(e)})

        # Keep your error-row schema exactly
        dec_counts_rows.append({"sample_id": sid, "DEC_count": np.nan})
        alv_counts_rows.append(
            {
                "sample_id": sid,
                "ALV_long_count": np.nan,
                "ALV_short_count": np.nan,
                "suggested": "error",
                "ALV_n_used": np.nan,
                "ALV_mean_size_used_px": np.nan,
                "ALV_surface_area_used_px": np.nan,
                "ALV_surface_area_used_frac": np.nan,
                "ALV_mean_size_used_mm2": np.nan,
                "ALV_surface_area_used_mm2": np.nan,
                "ALV_mean_size_long_px": np.nan,
                "ALV_surface_area_long_px": np.nan,
                "ALV_surface_area_long_frac": np.nan,
                "ALV_mean_size_long_mm2": np.nan,
                "ALV_surface_area_long_mm2": np.nan,
                "ALV_mean_size_short_px": np.nan,
                "ALV_surface_area_short_px": np.nan,
                "ALV_surface_area_short_frac": np.nan,
                "ALV_mean_size_short_mm2": np.nan,
                "ALV_surface_area_short_mm2": np.nan,
                "image_h_px": np.nan,
                "image_w_px": np.nan,
            }
        )

    progress.progress((i + 1) / len(orig_names))

progress.empty()
status.empty()

# Assemble dataframes (same)
dec_counts_df = pd.DataFrame(dec_counts_rows).sort_values("sample_id").reset_index(drop=True)
dec_particles_df = pd.concat(dec_particles_rows, ignore_index=True) if dec_particles_rows else pd.DataFrame()

alv_counts_df = pd.DataFrame(alv_counts_rows).sort_values("sample_id").reset_index(drop=True)
alv_particles_df = pd.concat(alv_particles_rows, ignore_index=True) if alv_particles_rows else pd.DataFrame()

errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["sample_id", "error"])

# Outputs always use original image names; no anonymisation swap needed.
export_name_map = None

# Display + downloads (same layout)
c1, c2 = st.columns(2)

with c1:
    st.subheader("Part I — DEC counts")
    st.dataframe(dec_counts_df, width="stretch")

    dec_sheets = {"DEC_counts": dec_counts_df}
    if include_particle_tables:
        dec_sheets["DEC_particles"] = dec_particles_df
    if not errors_df.empty:
        dec_sheets["errors"] = errors_df

    dec_excel = to_excel_bytes(dec_sheets)

    st.download_button(
        "Download DEC Excel",
        data=dec_excel,
        file_name="DEC_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with c2:
    st.subheader("Part II — Alveolar counts (Long vs Short)")
    st.dataframe(alv_counts_df, width="stretch")

    alv_sheets = {"ALV_counts": alv_counts_df}
    if include_particle_tables:
        alv_sheets["ALV_particles"] = alv_particles_df
    if not errors_df.empty:
        alv_sheets["errors"] = errors_df

    alv_excel = to_excel_bytes(alv_sheets)

    st.download_button(
        "Download Alveolar Excel",
        data=alv_excel,
        file_name="ALV_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

if not errors_df.empty:
    st.warning(f"{len(errors_df)} image(s) failed.")
    st.dataframe(errors_df, width="stretch")

# QC preview (same)
st.divider()
st.subheader(f"QC preview (first {len(qc_previews)} samples)")
st.caption("Left→Right: Original, DEC detected, ALV long, ALV short")

for sid, orig_j, dec_j, long_j, short_j in qc_previews:
    st.markdown(f"**{sid}**")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        if orig_j:
            st.image(orig_j, caption="Original", width="stretch")
    with r2:
        if dec_j:
            st.image(dec_j, caption="DEC overlay", width="stretch")
    with r3:
        if long_j:
            st.image(long_j, caption="ALV long overlay", width="stretch")
    with r4:
        if short_j:
            st.image(short_j, caption="ALV short overlay", width="stretch")

# Bundle ZIP (same)
bundle = make_bundle_zip(
    dec_excel=dec_excel,
    alv_excel=alv_excel,
    qc_items=qc_previews,
    params=run_params,
    errors_df=errors_df,
    name_map_df=export_name_map,
)

st.download_button(
    "Download ALL outputs as ZIP (Excels + QC + params + errors)",
    data=bundle,
    file_name="dec_alv_outputs_bundle.zip",
    mime="application/zip",
    width="stretch",
)

st.divider()
st.subheader("Protocol notes")
st.markdown(
    """
- **Mode cutoff** is the main knob for DEC (lower = stricter).
- DEC filters mimic your ImageJ macro:
  - size + circularity
  - delete if Mode > cutoff
  - delete if Round < round_min
  - delete if AR > ar_max
- Alveolar: runs both long/short and suggests one, but QC overlay is the real check.
- ALV summary fields:
  - `*_px` are raw pixel-area units (traceability)
  - `*_frac` is surface-area fraction (sum area / image area)
  - `*_mm2` require Calibration µm/px and report physical mm²
- If you enable **Fiji/ImageJ**, ALV counts/areas come from headless Fiji; overlays still come from the Python pipeline for consistent QC bundle images.
"""
)
