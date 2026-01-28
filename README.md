# Lung Histology — Alveolar airspaces + wall thickness

This folder contains a copy of the privacy-focused lung histology analysis tool developed for measuring alveolar airspaces and wall thickness.

Features
- Streamlit app for interactive segmentation and QC
- CLI for batch processing and exporting anonymized outputs
- Watershed-based alveoli separation with configurable marker strategies
- Exports CSV, NPY thickness maps and optional QC images
- Privacy-first defaults: anonymized outputs, no persistent uploads

Quick start (recommended: create Python virtual environment first)

1. Create and activate venv (macOS / Linux):

   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run Streamlit app:

   streamlit run testlh.py

4. Run CLI example (outputs saved to `out`):

   python testlh.py path/to/image.png --out_dir out --save_images --alveoli_overlay

Files
- `testlh.py`: main privacy-first script (Streamlit + CLI)
- `requirements.txt`: pinned dependencies
- `run_cli.sh`: example CLI command
- `LICENSE`: MIT license
- `.gitignore`: ignores venv and output folders

Notes
- The app uses scikit-image, OpenCV, NumPy, Pandas, SciPy and Streamlit.
- Default behaviour anonymizes outputs; pass `--keep_names` to the CLI to preserve filenames.

Contact
- For issues or feature requests, add an issue to the repository.
