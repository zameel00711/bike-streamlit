import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from bike_sharing import (
    CV_PATH,
    FEATURES,
    MODEL_NAMES,
    MODEL_PATH,
    OUTPUT_DIR,
    PLOT_DIR,
    TARGET,
    load_dataset,
    make_models,
)


RANDOM_STATE = 42


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def save_plots(
    data: pd.DataFrame,
    cv_results: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    best_name: str,
    best_pipeline,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=data, x=TARGET, bins=45, kde=True, ax=ax, color="#2878B5")
    ax.set(title="Hourly bike rental demand distribution", xlabel="Bike rentals per hour")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "demand_distribution.png", dpi=180)
    plt.close(fig)

    corr_columns = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        TARGET,
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        data[corr_columns].corr(numeric_only=True),
        cmap="vlag",
        center=0,
        annot=False,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation heatmap of features and hourly rentals")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.35, s=15, color="#2878B5")
    axis_min = min(float(y_test.min()), float(predictions.min()))
    axis_max = max(float(y_test.max()), float(predictions.max()))
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "--", color="#D9534F")
    ax.set(
        title=f"Observed vs predicted hourly rentals ({best_name})",
        xlabel="Observed rentals",
        ylabel="Predicted rentals",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "observed_vs_predicted.png", dpi=180)
    plt.close(fig)

    ordered = cv_results.sort_values("CV RMSE")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        ordered["Model"],
        ordered["CV RMSE"],
        yerr=ordered["CV RMSE std"],
        capsize=5,
        color=["#2878B5" if name == best_name else "#9BBBD1" for name in ordered["Model"]],
    )
    ax.set(
        title="Five-fold cross-validation model comparison",
        ylabel="Mean validation RMSE (lower is better)",
    )
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    regressor = best_pipeline.named_steps["regressor"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if hasattr(regressor, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importances = pd.Series(regressor.feature_importances_, index=feature_names)
        top_features = importances.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        top_features.plot.barh(ax=ax, color="#2878B5")
        ax.set(title="Most influential model features", xlabel="Random forest importance")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "feature_importance.png", dpi=180)
        plt.close(fig)


def create_report(
    data: pd.DataFrame,
    test_results: pd.DataFrame,
    cv_results: pd.DataFrame,
    best_name: str,
    best_cv_rmse: float,
) -> Path:
    report_path = OUTPUT_DIR / "bike_sharing_report.pdf"
    metrics = test_results.set_index("Model")
    winner = metrics.loc[best_name]
    missing_count = int(data[FEATURES + [TARGET]].isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    median_demand = float(data[TARGET].median())

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Bike Sharing Demand Prediction", fontsize=23, weight="bold")
        fig.text(0.08, 0.905, "Random forest and regression model comparison", fontsize=13)
        page_one = (
            f"Problem statement\n"
            f"Estimate the number of bicycle rentals in an hour from calendar, weather, "
            f"and normalized weather measurements. A reliable estimate can help a "
            f"bike-share operator anticipate hourly fleet demand.\n\n"
            f"Dataset\n"
            f"The UCI Bike Sharing Dataset (hour.csv) contains {len(data):,} hourly "
            f"records. It provides 12 predictors used here and the numeric target "
            f"cnt (rental count). The target median is {median_demand:.0f} rentals "
            f"per hour. The original casual and registered counts are deliberately "
            f"excluded because they are components of cnt and would leak the answer.\n\n"
            f"Preprocessing\n"
            f"Missing values among selected columns: {missing_count}; duplicate rows: "
            f"{duplicates}. Numeric weather inputs are median-imputed and standardized. "
            f"Calendar and weather-category codes are imputed with the most frequent "
            f"value and one-hot encoded. Preprocessing is inside each model Pipeline, "
            f"so each validation fold learns transformations from its training fold.\n\n"
            f"Training and model selection\n"
            f"Data was divided 80/20 with random_state={RANDOM_STATE}. The models are "
            f"Random Forest (nonlinear tree ensemble) and Ridge Regression "
            f"(regularized linear baseline). "
            f"Five-fold shuffled cross-validation on the training set selected the "
            f"model with the lowest mean RMSE: {best_name} ({best_cv_rmse:.2f} rentals). "
            f"The test set was reserved for final evaluation."
        )
        fig.text(0.08, 0.85, page_one, fontsize=10.5, va="top", linespacing=1.5, wrap=True)
        fig.text(
            0.08,
            0.08,
            "Dataset source: UCI Machine Learning Repository, Bike Sharing Dataset "
            "(Fanaee-T and Gama, 2013).",
            fontsize=9,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Evaluation results", fontsize=21, weight="bold")
        table_data = []
        for _, row in test_results.iterrows():
            cv_row = cv_results.loc[cv_results["Model"] == row["Model"]].iloc[0]
            table_data.append(
                [
                    row["Model"],
                    f'{cv_row["CV RMSE"]:.2f}',
                    f'{row["MAE"]:.2f}',
                    f'{row["RMSE"]:.2f}',
                    f'{row["R2"]:.3f}',
                ]
            )
        ax = fig.add_axes([0.08, 0.70, 0.84, 0.18])
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            colLabels=["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        fig.text(
            0.08,
            0.64,
            f"Selected model: {best_name}\n"
            f"Test MAE: {winner['MAE']:.2f} rentals; test RMSE: {winner['RMSE']:.2f}; "
            f"test R²: {winner['R2']:.3f}.\n\n"
            f"MAE is typical absolute error; RMSE penalizes large misses and is\n"
            f"the cross-validation selection measure. R² measures test-set fit.\n"
            f"Classification scores do not apply to regression.",
            fontsize=9.5,
            va="top",
            linespacing=1.35,
        )
        image = plt.imread(PLOT_DIR / "demand_distribution.png")
        ax = fig.add_axes([0.12, 0.04, 0.76, 0.20])
        ax.imshow(image)
        ax.axis("off")
        fig.text(0.13, 0.015, "Figure 1. Distribution of hourly rental counts.", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Patterns and predictions", fontsize=21, weight="bold")
        for axes, filename, caption, bounds in [
            (
                [0.10, 0.54, 0.80, 0.32],
                "correlation_heatmap.png",
                "Figure 2. Correlations show pairwise linear associations; they do not imply causation.",
                0.51,
            ),
            (
                [0.14, 0.10, 0.72, 0.34],
                "observed_vs_predicted.png",
                "Figure 3. Points near the diagonal indicate closer predictions; spread indicates error.",
                0.07,
            ),
        ]:
            ax = fig.add_axes(axes)
            ax.imshow(plt.imread(PLOT_DIR / filename))
            ax.axis("off")
            fig.text(0.10, bounds, caption, fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Analysis, limitations, and course outcome", fontsize=19, weight="bold")
        analysis = (
            f"Model comparison\n"
            f"{best_name} had the lowest mean five-fold validation RMSE ({best_cv_rmse:.2f}) "
            f"and was selected without using test-set scores. Model-comparison graph error "
            f"bars show variation in RMSE across the five folds. The selected model's test "
            f"error is {winner['RMSE']:.2f} rentals RMSE and {winner['MAE']:.2f} rentals MAE.\n\n"
            f"Understanding errors\n"
            f"An underprediction can leave too few bicycles available; an overprediction can "
            f"leave bikes underused. The costs depend on station, time, and rebalancing "
            f"capacity. For operations, MAE is an intuitive typical error, while RMSE is "
            f"useful when large demand misses are especially costly. A real deployment "
            f"should also evaluate peak-hour or station-specific errors and choose metrics "
            f"based on operational costs.\n\n"
            f"Limitations\n"
            f"This is historical data from a limited period and region; demand patterns "
            f"can change. The dataset has no station-level availability, event, service "
            f"disruption, or real-time information. Random splitting can place nearby "
            f"hours in both train and test sets, so it may overstate performance on future "
            f"unseen time periods. A chronological holdout and external validation are "
            f"recommended before operational use. Predictions are estimates, not guarantees.\n\n"
            f"CO5 – Solve real life problems using appropriate machine learning models "
            f"and evaluate the performance measures\n"
            f"This project translates hourly bike allocation into a supervised regression "
            f"problem, compares suitable models using five-fold cross-validation, evaluates "
            f"the selected approach on a held-out test set, visualizes the results, and "
            f"discusses how prediction errors affect the real-world decision."
        )
        fig.text(0.08, 0.86, analysis, fontsize=10, va="top", linespacing=1.5, wrap=True)
        ax = fig.add_axes([0.14, 0.04, 0.72, 0.22])
        ax.imshow(plt.imread(PLOT_DIR / "model_comparison.png"))
        ax.axis("off")
        fig.text(0.10, 0.02, "Figure 4. Five-fold cross-validation RMSE; lower is better.", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

    return report_path


def main() -> None:
    data = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"Missing values in selected columns: {int(data[FEATURES + [TARGET]].isna().sum().sum())}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    demand_groups = pd.cut(data[TARGET], bins=5, include_lowest=True)
    print("Demand-range counts (descriptive only, not classification labels):")
    print(demand_groups.value_counts().sort_index().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_rows = []
    cv_rows = []
    fitted_models = {}
    for name, pipeline in make_models().items():
        cv_scores = -cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        test_rows.append({"Model": name, **regression_metrics(y_test, prediction)})
        cv_rows.append(
            {
                "Model": name,
                "CV RMSE": float(cv_scores.mean()),
                "CV RMSE std": float(cv_scores.std()),
            }
        )
        fitted_models[name] = pipeline

    test_results = pd.DataFrame(test_rows)
    cv_results = pd.DataFrame(cv_rows)
    test_results.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    cv_results.to_csv(CV_PATH, index=False)

    best_name = cv_results.sort_values("CV RMSE").iloc[0]["Model"]
    best_cv_rmse = float(cv_results.loc[cv_results["Model"] == best_name, "CV RMSE"].iloc[0])
    selected_pipeline = fitted_models[best_name]
    selected_test_prediction = selected_pipeline.predict(x_test)

    save_plots(
        data,
        cv_results,
        y_test,
        selected_test_prediction,
        best_name,
        selected_pipeline,
    )

    final_pipeline = make_models()[best_name]
    final_pipeline.fit(data[FEATURES], data[TARGET])
    joblib.dump(final_pipeline, MODEL_PATH)
    metadata = {
        "selected_model": best_name,
        "model_key": MODEL_NAMES[best_name],
        "cv_rmse": best_cv_rmse,
        "test_metrics": test_results.set_index("Model").loc[best_name].to_dict(),
        "features": FEATURES,
        "dataset_rows": len(data),
        "dataset_source": "UCI Bike Sharing Dataset, hourly data",
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report_path = create_report(data, test_results, cv_results, best_name, best_cv_rmse)

    print("\nHeld-out test results:")
    print(test_results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected by five-fold CV RMSE: {best_name} ({best_cv_rmse:.3f})")
    print(f"Saved final model: {MODEL_PATH}")
    print(f"Saved report: {report_path}")
    print(f"Saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
