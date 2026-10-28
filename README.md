# Bike sharing demand prediction

A complete machine-learning assignment project using the UCI Bike Sharing Dataset. It trains and compares regression models, selects a model with five-fold cross-validation, saves evaluation plots and an A4 PDF report, and provides an interactive Streamlit prediction app.

## Project contents

- `data/hour.csv` — hourly UCI Bike Sharing Dataset.
- `bike_sharing.py` — shared data loading, preprocessing, and model definitions.
- `train_model.py` — split, five-fold model selection, test-set evaluation, plots, saved model, and PDF report.
- `streamlit_app.py` — form-based interactive demand predictor.
- `requirements.txt` — Python dependencies.
- `outputs/` — generated artifacts after training (ignored by Git).

## Setup

Use Python 3.10 or newer. In a terminal opened in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The included hourly data file comes from the [UCI Machine Learning Repository Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). If it is missing, download the dataset ZIP from UCI and place its `hour.csv` at `data/hour.csv`.

## Train, evaluate, and create the report

```powershell
python train_model.py
```

Training prints the data shape, column names, missing-value/duplicate checks, and descriptive target ranges. Outputs are written to `outputs/`:

- `best_model.joblib` and `model_metadata.json`
- `test_metrics.csv` and `cross_validation.csv`
- `bike_sharing_report.pdf` (four A4 pages)
- `plots/` with demand distribution, correlation heatmap, observed-vs-predicted chart, model comparison, and (for tree models) feature importance

The 80/20 split uses `random_state=42`. This is a regression problem, so classification-only measures (accuracy, precision, recall, ROC-AUC, confusion matrix, and stratified splitting) are not appropriate. Random Forest is compared with Ridge Regression, a regularized linear baseline. Models are ranked by mean five-fold training-set RMSE; the test set is held out for final evaluation. Metrics reported are MAE, RMSE, and R².

Preprocessing is part of each scikit-learn Pipeline: numeric values are median-imputed and standardized, while categorical calendar/weather fields are imputed and one-hot encoded. `casual` and `registered` are excluded because they leak the target `cnt`.

## Run the Streamlit app

Train the model first, then run:

```powershell
streamlit run streamlit_app.py
```

Enter hourly calendar and weather conditions and choose **Predict demand**. The weather measurements use the dataset's normalized 0–1 scale. The estimate represents total rentals in one hour; it does not estimate station-level availability.

## Dataset citation

Fanaee-T, H. and Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. UCI Machine Learning Repository, Bike Sharing Dataset.
