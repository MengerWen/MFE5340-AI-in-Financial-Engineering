# Graph-Enhanced Conditional Latent Factor Pricing

This project lives inside the repository subdirectory `Final Project/`. Keep project code, configs, outputs, and reports inside this folder unless there is an explicit reason to touch another assignment directory.

## Environment

Use the Anaconda interpreter that contains the project stack:

```powershell
$PY = "d:\MG\anaconda3\python.exe"
```

Do not use the system `python` for this project unless it points to the same Anaconda environment. The current code assumes the scientific stack plus the modeling stack are available: pandas, numpy, scikit-learn, scipy, statsmodels, linearmodels, PyYAML, PyTables, h5py, matplotlib, seaborn, networkx, pyarrow, cvxpy, torch, torchvision, torchaudio, torch_geometric, tensorboard, and einops.

`requirements.txt` records the core versions observed in this Anaconda environment. Treat it as reproducibility documentation unless you intentionally rebuild the environment.

## Structure

- `src/data/`: pandas-based data loading, inspection, and monthly panel construction.
- `src/features/`: pandas-exportable feature manifests.
- `src/models/`: benchmark registry plus torch neural model skeletons for MLP, conditional beta MLP, GCN, and GAT.
- `src/graphs/`: pandas/sklearn graph construction, networkx conversion, and PyG `Data` conversion.
- `src/training/`: rolling/expanding OOS protocol helpers, torch seed/device setup, and TensorBoard logging helpers.
- `src/evaluation/`: pandas/numpy/scikit-learn metrics and linearmodels panel pricing diagnostics.
- `src/portfolio/`: pandas/numpy portfolio weight and turnover helpers.
- `configs/`: experiment and cleaning configuration files.
- `scripts/`: runnable entry points.
- `outputs/`: generated artifacts, ignored by git by default.
- `reports/`: protocol and audit reports.

Existing `Guideline.md`, `Proposal.md`, and raw `data/` files are preserved.

## Recommended Commands

Run these from `Final Project/`:

```powershell
& "d:\MG\anaconda3\python.exe" scripts/check_environment.py
& "d:\MG\anaconda3\python.exe" scripts/load_data.py
& "d:\MG\anaconda3\python.exe" scripts/inspect_data.py
& "d:\MG\anaconda3\python.exe" scripts/build_panel.py --config configs/cleaning_features500.yaml
& "d:\MG\anaconda3\python.exe" scripts/evaluate.py
& "d:\MG\anaconda3\python.exe" scripts/list_benchmarks.py
& "d:\MG\anaconda3\python.exe" scripts/build_graph.py
& "d:\MG\anaconda3\python.exe" scripts/train.py
& "d:\MG\anaconda3\python.exe" scripts/backtest_portfolio.py
```

`inspect_data.py` and `build_panel.py` are the main Stage 2 data commands. `build_graph.py`, `train.py`, and `backtest_portfolio.py` currently validate configs and expose reusable helper modules; full graph modeling, neural training, and portfolio backtesting are reserved for later stages.

The panel builder writes:

- `outputs/metadata/data_inspection_stage2.json`
- `outputs/panels/main_features500_panel.pkl`
- `outputs/metadata/main_features500_panel_metadata.json`
- `reports/data_audit_stage2.md`

## Panel Logic

The main-spec panel uses `features500/`, CSI 500 membership at month `t`, and next-month returns from `monthly_returns.pkl` as the target. `target_excess_return` subtracts the compounded daily risk-free return over month `t + 1`.

Feature cleaning uses pandas/numpy operations by month: robust clipping, cross-sectional median imputation, and cross-sectional normalization. Graph utilities can convert return-correlation kNN edges into networkx graphs and torch_geometric `Data` objects. Torch utilities set reproducible seeds, select CUDA when available, and prepare TensorBoard logging.
