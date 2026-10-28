from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "hour.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
CV_PATH = OUTPUT_DIR / "cross_validation.csv"

TARGET = "cnt"
CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MODEL_NAMES = {
    "Random forest": "random_forest",
    "Ridge regression": "ridge",
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. See the README for the UCI download."
        )
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    model_specs = {
        "Random forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge regression": Ridge(alpha=10.0),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("regressor", estimator),
            ]
        )
        for name, estimator in model_specs.items()
    }
