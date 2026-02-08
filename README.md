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
