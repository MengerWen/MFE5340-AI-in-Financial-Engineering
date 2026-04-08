# Graph-Enhanced Conditional Latent Factor Pricing

This project lives inside the repository subdirectory `Final Project/`. Keep project code, configs, outputs, and reports inside this folder unless there is an explicit reason to touch another assignment directory.

## Environment

Use the Anaconda interpreter that already contains the required scientific Python stack:

```powershell
$PY = "d:\MG\anaconda3\python.exe"
```

Do not use the system `python` for this project unless it is explicitly pointed to the same Anaconda environment. The code now assumes pandas, numpy, scikit-learn, PyYAML, PyTables, h5py, scipy, statsmodels, matplotlib, seaborn, networkx, pyarrow, and cvxpy are available.

`requirements.txt` records the core versions observed in this Anaconda environment. It is documentation for reproducibility, not a request to install packages into another Python unless needed.

## Structure

- `src/data/`: pandas-based data loading, inspection, and monthly panel construction.
- `src/features/`: pandas-exportable feature manifests.
- `src/models/`: pandas/scikit-learn-style benchmark registry and estimator interface.
- `src/graphs/`: graph construction specifications for later stages.
- `src/training/`: rolling/expanding OOS protocol stubs.
- `src/evaluation/`: pandas/numpy/scikit-learn metric helpers.
- `src/portfolio/`: portfolio backtest configuration stubs.
- `configs/`: experiment and cleaning configuration files.
- `scripts/`: runnable entry points.
- `outputs/`: generated artifacts, ignored by git by default.
- `reports/`: protocol and audit reports.

Existing `Guideline.md`, `Proposal.md`, and raw `data/` files are preserved.

## Stage 2 Commands

Run these from `Final Project/`:

```powershell
& "d:\MG\anaconda3\python.exe" scripts/load_data.py
& "d:\MG\anaconda3\python.exe" scripts/inspect_data.py
& "d:\MG\anaconda3\python.exe" scripts/build_panel.py --config configs/cleaning_features500.yaml
& "d:\MG\anaconda3\python.exe" scripts/evaluate.py
& "d:\MG\anaconda3\python.exe" scripts/list_benchmarks.py
```

The panel builder writes:

- `outputs/metadata/data_inspection_stage2.json`
- `outputs/panels/main_features500_panel.pkl`
- `outputs/metadata/main_features500_panel_metadata.json`
- `reports/data_audit_stage2.md`

## Panel Logic

The main-spec panel uses `features500/`, CSI 500 membership at month `t`, and next-month returns from `monthly_returns.pkl` as the target. `target_excess_return` subtracts the compounded daily risk-free return over month `t + 1`.

Feature cleaning uses pandas/numpy operations by month: robust clipping, cross-sectional median imputation, and cross-sectional normalization. Later modeling stages should keep using `d:\MG\anaconda3\python.exe`.
