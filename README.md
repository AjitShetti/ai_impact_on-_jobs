# AI Impact on Jobs (2010–2025)

A small ML project that analyzes the impact of AI on jobs using tabular data and CatBoost.

**Contents**
- **Project:** data processing, training, and prediction pipelines.
- **Data:** located in the `data/` folder.
- **Notebooks:** exploratory and training notebooks under `notebook/`.

**Quick Start (Windows)**

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run notebooks for EDA or model training interactively:

- Open `notebook/01_EDA.ipynb` and `notebook/02_model_trainer.ipynb` in Jupyter.

3. Run the training pipeline (scripted):

```powershell
python src/pipeline/train_pipeline.py
```

4. Run prediction / inference pipeline:

```powershell
python src/pipeline/predict_pipeline.py
```

5. Or run the main app (if provided):

```powershell
python app.py
```

Project layout (important files)
- `app.py` — project entrypoint for quick runs / inference (if implemented).
- `requirements.txt` — Python dependencies.
- `data/` — input datasets (e.g. `ai_impact_jobs_2010_2025.csv`).
- `artifacts/` — produced CSVs and model artifacts (`train.csv`, `test_csv`, `raw.csv`).
- `notebook/` — Jupyter notebooks for EDA and model training.
- `src/` — source package with pipeline components:
	- `src/components/data_ingestion.py`
	- `src/components/data_transformation.py`
	- `src/components/model_trainer.py`
	- `src/pipeline/train_pipeline.py`
	- `src/pipeline/predict_pipeline.py`

Data & artifacts
- Raw dataset: `data/ai_impact_jobs_2010_2025.csv`.
- Training artifacts and CatBoost outputs are in `artifacts/` and `catboost_info/` after training.

Notes
- The repository uses CatBoost for modeling (see `catboost_info/` for training logs).
- If a script accepts configuration or paths, edit the script headers or call from notebooks as needed.

Contributing
- For fixes or improvements, open a PR with focused changes. Add tests if applicable.

License
- Add or update a license file if you plan to open-source this repository.

Contact
- For questions about this project, update the `README.md` with maintainer contact info.
