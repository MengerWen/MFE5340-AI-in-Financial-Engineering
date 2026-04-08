# Graph-Enhanced Conditional Latent Factor Pricing

This project lives inside the repository subdirectory `Final Project/`. Keep project code, configs, outputs, and reports inside this folder unless there is an explicit reason to touch another assignment directory.

## Stage 1 Contents

Stage 1 sets up a research-oriented skeleton only. It does not train models or build the final panel.

Key folders:

- `src/data/`: data location and preprocessing interfaces.
- `src/features/`: feature manifest helpers.
- `src/models/`: benchmark model registry and estimator interface.
- `src/graphs/`: graph construction specification stubs.
- `src/training/`: OOS protocol stubs.
- `src/evaluation/`: planned metric names and small metric helpers.
- `src/portfolio/`: portfolio backtest configuration stubs.
- `configs/`: experiment configuration files.
- `scripts/`: runnable entry points for later stages.
- `outputs/`: generated artifacts, ignored by git by default.
- `reports/`: protocol and data inventory notes.

Existing `Guideline.md`, `Proposal.md`, and `data/` files are preserved.

## Environment

From `Final Project/`, install dependencies in a project environment:

```powershell
python -m pip install -r requirements.txt
```

The active Stage 1 shell had `python` available but did not have `pandas` installed, so pickle and HDF5 schemas were not loaded yet.

## Useful Stage 1 Commands

Run these from `Final Project/`:

```powershell
python scripts/load_data.py
python scripts/run_preprocessing.py
python scripts/list_benchmarks.py
python scripts/build_graph.py
python scripts/train.py
python scripts/evaluate.py
python scripts/backtest_portfolio.py
```

These commands validate structure and defaults only. Later stages should replace the placeholders with point-in-time panel construction, graph snapshots, model training, evaluation tables, and portfolio reports.

## Next Stage

Stage 2 should begin by installing the environment, loading each raw data object, documenting exact schemas, and building a reproducible point-in-time monthly panel for the `features500/` main specification.

## Stage 2 Data Panel

Use the Anaconda interpreter that has the data stack installed:

```powershell
& "d:\MG\anaconda3\python.exe" scripts/inspect_data.py
& "d:\MG\anaconda3\python.exe" scripts/build_panel.py --config configs/cleaning_features500.yaml
```

Stage 2 outputs:

- `outputs/metadata/data_inspection_stage2.json`
- `outputs/panels/main_features500_panel.pkl`
- `outputs/metadata/main_features500_panel_metadata.json`
- `reports/data_audit_stage2.md`

The main-spec panel uses `features500/`, CSI 500 membership at month `t`, and next-month returns from `monthly_returns.pkl` as the target.
