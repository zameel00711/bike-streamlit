import json

import joblib
import pandas as pd
import streamlit as st

from bike_sharing import FEATURES, MODEL_PATH, OUTPUT_DIR


st.set_page_config(
    page_title="Bike sharing demand",
    page_icon="🚲",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained model is missing. Run `python train_model.py` before starting the app."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    metadata_path = OUTPUT_DIR / "model_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Model metadata is missing. Run `python train_model.py` before starting the app."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


st.title("Bike sharing demand prediction")
st.write(
    "Estimate hourly bike rentals from calendar and weather conditions using a "
    "model selected by five-fold cross-validation."
)

try:
    model = load_model()
    metadata = load_metadata()
except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.stop()

st.caption(
    f"Selected model: {metadata['selected_model']} · "
    f"Cross-validation RMSE: {metadata['cv_rmse']:.1f} rentals"
)

season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
weather_names = {
    1: "Clear / partly cloudy",
    2: "Mist / cloudy",
    3: "Light snow or rain",
    4: "Heavy rain / snow",
}
weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

with st.form("demand_inputs"):
    st.subheader("Hourly conditions")
    first_row = st.columns(3)
    season_label = first_row[0].selectbox(
        "Season", options=list(season_names.values()), index=1
    )
    year_label = first_row[1].selectbox("Year", options=[2011, 2012], index=1)
    month = first_row[2].select_slider("Month", options=list(range(1, 13)), value=6)

    second_row = st.columns(3)
    hour = second_row[0].slider("Hour of day", min_value=0, max_value=23, value=8)
    weekday = second_row[1].selectbox(
        "Day of week", options=list(range(7)), format_func=lambda day: weekday_names[day]
    )
    weather_label = second_row[2].selectbox(
        "Weather", options=list(weather_names.values()), index=0
    )

    third_row = st.columns(3)
    temperature = third_row[0].slider(
        "Temperature (normalized)", 0.0, 1.0, 0.5, 0.01,
        help="Use the dataset's normalized temperature scale: 0 is coldest, 1 is hottest.",
    )
    feels_like = third_row[1].slider(
        "Feels-like temperature (normalized)", 0.0, 1.0, 0.5, 0.01
    )
    humidity = third_row[2].slider(
        "Humidity (normalized)", 0.0, 1.0, 0.5, 0.01
    )

    fourth_row = st.columns(3)
    windspeed = fourth_row[0].slider(
        "Windspeed (normalized)", 0.0, 1.0, 0.2, 0.01
    )
    holiday = fourth_row[1].checkbox("Public holiday")
    workingday = fourth_row[2].checkbox("Working day", value=True)
    submitted = st.form_submit_button("Predict demand", type="primary")

if submitted:
    season_code = next(code for code, name in season_names.items() if name == season_label)
    weather_code = next(code for code, name in weather_names.items() if name == weather_label)
    input_row = {
        "season": season_code,
        "yr": year_label - 2011,
        "mnth": month,
        "hr": hour,
        "holiday": int(holiday),
        "weekday": weekday,
        "workingday": int(workingday),
        "weathersit": weather_code,
        "temp": temperature,
        "atemp": feels_like,
        "hum": humidity,
        "windspeed": windspeed,
    }
    prediction = max(
        0.0,
        float(model.predict(pd.DataFrame([input_row], columns=FEATURES))[0]),
    )
    st.metric("Estimated bike rentals", f"{prediction:,.0f}")
    st.caption(
        "This is an estimate of total rentals for one hour. It is based on historical "
        "regional data and does not account for station capacity or live availability."
    )

with st.expander("About the dataset and evaluation"):
    st.write(
        "The app uses the hourly UCI Bike Sharing Dataset. `casual` and `registered` "
        "counts are excluded from model inputs because they directly add up to the "
        "rental-count target. See the generated A4 report for the test metrics, "
        "cross-validation comparison, graphs, and limitations."
    )
    plot_files = [
        ("demand_distribution.png", "Hourly rental-count distribution"),
        ("correlation_heatmap.png", "Feature correlation heatmap"),
        ("observed_vs_predicted.png", "Observed and predicted rentals"),
        ("model_comparison.png", "Five-fold model comparison"),
        ("feature_importance.png", "Selected model feature importance"),
    ]
    available_plots = [
        (name, label) for name, label in plot_files if (OUTPUT_DIR / "plots" / name).is_file()
    ]
    if available_plots:
        for name, label in available_plots:
            st.image(OUTPUT_DIR / "plots" / name, caption=label, alt=label, width="stretch")
    else:
        st.info("Run `python train_model.py` to generate the evaluation plots.")

    report_path = OUTPUT_DIR / "bike_sharing_report.pdf"
    if report_path.is_file():
        st.download_button(
            "Download the A4 project report",
            data=report_path.read_bytes(),
            file_name="bike_sharing_report.pdf",
            mime="application/pdf",
        )
