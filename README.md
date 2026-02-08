# Lung Histology — DEC + Alveolar Batch Counter

This folder contains a Streamlit app for batch quantification of:
- **DEC** particles (counting + optional particle table)
- **Alveolar (ALV)** airspace particles using **long** and **short** variants (counts + area summaries)

It is designed for **batch processing with QC previews** and **exportable Excel + ZIP bundles**.

## Features
- Upload **one/more images** (`.tif/.tiff/.jpg/.jpeg/.png`) or a **ZIP** of images
- Optional **EXIF/metadata stripping** on ingest (recommended)
- Runs **DEC** + **ALV (long & short)** for each image
- Heuristic suggestion for whether **long** or **short** segmentation is more plausible
- Optional **particle tables** (larger outputs, slower)
- Optional **Fiji/ImageJ headless** execution for ALV counting (macOS)
- Exports:
  - `DEC_results.xlsx`
  - `ALV_results.xlsx`
  - `dec_alv_outputs_bundle.zip` (Excels + QC overlays + params + errors)

## Quick start

### 1) Create + activate a virtual environment (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the Streamlit app

```bash
streamlit run histology_app.py
```

## How to use (recommended workflow)
1. Upload images (or a ZIP).
2. Leave **Strip EXIF/metadata** enabled unless you explicitly need embedded metadata.
3. Start with default parameters and run on a small subset first.
4. Inspect the **QC preview** overlays:
   - *Original* (reference)
   - *DEC detected* (green overlay + contours)
   - *ALV long* and *ALV short* (green overlay + contours)
5. Adjust parameters, rerun, and confirm overlays improve.
6. When satisfied, rerun on the full dataset and download the Excel/ZIP outputs.

## Parameter tuning notes

### Global / output options
- **Include particle tables**: adds `DEC_particles` and `ALV_particles` sheets.
  - Use this when you need traceability per detected object.
  - Leave off for large batches (files get big; processing is slower).
- **QC preview count**: how many samples get overlay images in the ZIP and shown in the UI.
  - Increase for spot-checking; set to `0` to disable QC image export.
- **QC JPEG quality**: smaller ZIP vs clearer overlays.

### DEC parameters (particle filtering)
DEC detection is edge-based and then filtered to keep particles that match expected morphology.
Use the **DEC overlay** as the primary guide.

- **Mode cutoff (delete if Mode > cutoff)**:
  - Main sensitivity knob.
  - Lower values = stricter filtering (removes brighter objects).
  - If you have obvious false positives, reduce the cutoff.
  - If true DEC are being removed, increase the cutoff.
- **Particle area min/max**:
  - Controls size range in pixels.
  - If noise speckles are included, increase *min*.
  - If true DEC are being missed, reduce *min* or increase *max* depending on what you see.
- **Circularity min/max**:
  - Higher circularity selects rounder objects.
  - If elongated fragments are included, increase *circularity min*.
  - If true DEC are not perfectly round, decrease *circularity min* slightly.
- **Round min (delete if Round < min)**:
  - Another “roundness” constraint; higher = stricter.
  - If objects look irregular but should count, reduce this.
- **AR max (delete if AR > max)**:
  - Aspect ratio constraint; lower = stricter against elongated objects.
  - If fibers/edges are being counted, reduce AR max.

### ALV parameters (segmentation + particle filtering)
ALV runs **both** variants:
- **Long**: includes an invert step (tends to swap foreground/background compared to short)
- **Short**: no invert

The app reports:
- `ALV_long_count`, `ALV_short_count`
- `suggested` (heuristic; use QC overlays to confirm)
- “used” fields where the app picks **short only if suggested==short; otherwise long**

Key parameters:
- **Alveolar min particle area**:
  - Filters out small regions.
  - If you’re counting tiny noise regions, increase this.
  - If true alveoli are excluded, decrease this.
- **Background subtract radius (rolling)**:
  - Larger values remove broader illumination gradients.
  - If shading causes broken segmentation, increase rolling.
  - If you lose local contrast/detail, decrease rolling.
- **Gaussian blur sigma**:
  - Smooths edge response.
  - If segmentation is too “salt and pepper”, increase sigma.
  - If boundaries get over-smoothed and merge, decrease sigma.
- **Contrast saturation % (approx)**:
  - Contrast stretch amount.
  - If edges are weak, increase slightly.
  - If everything becomes high-contrast and noisy, reduce.

### Calibration (optional)
- **Microns per pixel (µm/px)** enables physical units for ALV:
  - `*_mm2` fields are derived from pixel area using `(µm/px)^2 * 1e-6`.
- Leave as `0` if you only want pixel-based outputs.

### Fiji/ImageJ (optional)
- Enable this if you need ALV counts/areas to match an ImageJ macro workflow more closely.
- The app still generates Python overlays for QC even when Fiji is used for counting.
- Make sure the Fiji executable is available and executable (macOS example: `/Applications/Fiji.app/ImageJ-macosx`).

## Inputs
- Single image(s): `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.png`
- ZIP file containing any mix of the supported image types

If a ZIP and standalone image(s) contain the same basename, later items override earlier ones.

## Outputs

### Excel outputs
- `DEC_results.xlsx`
  - `DEC_counts` (always)
  - `DEC_particles` (if enabled)
  - `errors` (if any)
- `ALV_results.xlsx`
  - `ALV_counts` (always)
  - `ALV_particles` (if enabled)
  - `errors` (if any)

### Bundle output
- `dec_alv_outputs_bundle.zip`
  - `DEC_results.xlsx`, `ALV_results.xlsx`
  - `params.json` (run configuration)
  - `errors.csv` (if any)
  - `qc/` overlays for a subset of samples:
    - `{sample_id}_original.jpg`
    - `{sample_id}_dec.jpg`
    - `{sample_id}_alv_long.jpg`
    - `{sample_id}_alv_short.jpg`

## Fiji/ImageJ (optional)
ALV counting can be delegated to Fiji/ImageJ in **headless** mode.

- macOS default locations are probed automatically
- or provide the executable path in the sidebar (e.g. `/Applications/Fiji.app/ImageJ-macosx`)

Even when Fiji is used for ALV counts/areas, the app still generates **Python-based QC overlays** for consistency.

## Files
- `histology_app.py`: Streamlit app (DEC + ALV batch counter)
- `requirements.txt`: Python dependencies
- `LICENSE`: project license (if present)

## Notes
- Calibration (µm/px) is optional. If provided, ALV mean size and surface area are also reported in **mm²**.
- Outputs use the **original filenames** as `sample_id` (no anonymisation step in this app).
