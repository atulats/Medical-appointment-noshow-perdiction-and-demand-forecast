"""Clinic-facing Streamlit application for no-show risk and demand forecasting."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from medical_pipeline import forecast_demand, load_model, predict_no_show


BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "models" / "no_show_classifier.joblib"
FORECASTER_PATH = BASE_DIR / "models" / "demand_forecaster.joblib"

st.set_page_config(
    page_title="Medical appointment operations",
    page_icon=":material/clinical_notes:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained models...")
def load_artifacts() -> tuple[dict, dict]:
    if not CLASSIFIER_PATH.exists() or not FORECASTER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_models.py` before starting the app."
        )
    return load_model(CLASSIFIER_PATH), load_model(FORECASTER_PATH)


def risk_label(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= max(0.20, threshold * 0.65):
        return "Moderate"
    return "Low"


st.title("Medical appointment operations")
st.caption(
    "Estimate no-show risk for an appointment and plan daily specialist capacity. "
    "Predictions support staff decisions and should be monitored after deployment."
)

try:
    classifier, forecaster = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(str(exc), icon=":material/error:")
    st.code("python train_models.py\nstreamlit run streamlit_app.py", language="bash")
    st.stop()

with st.container(horizontal=True):
    st.metric(
        "Classifier",
        classifier["model_name"].replace("_", " ").title(),
        border=True,
    )
    st.metric(
        "Test F1",
        f"{classifier['test_metrics']['f1']:.3f}",
        border=True,
    )
    st.metric(
        "Test ROC-AUC",
        f"{classifier['test_metrics']['roc_auc']:.3f}",
        border=True,
    )
    st.metric(
        "Forecast MAPE",
        f"{forecaster['test_metrics']['mape']:.1%}",
        border=True,
    )

no_show_tab, forecast_tab = st.tabs(
    ["No-show predictor", "Demand forecaster"]
)

with no_show_tab:
    st.header("No-show risk")
    st.write(
        "Enter the information available before the appointment. The score is intended "
        "to prioritize reminders, not to deny or delay care."
    )

    with st.form("no_show_form", border=True):
        demographic_col, appointment_col, clinical_col = st.columns(3)
        with demographic_col:
            age = st.number_input("Age", min_value=0, max_value=110, value=18)
            gender = st.selectbox("Gender", ["F", "M", "I"])
            disability = st.selectbox(
                "Disability", ["intellectual", "motor", "unknown"]
            )
            needs_companion = st.checkbox("Patient needs a companion")
            place = st.selectbox(
                "Municipality",
                [
                    "ITAJAÍ",
                    "B. CAMBORIU",
                    "CAMBORIU",
                    "NAVEGANTES",
                    "ITAPEMA",
                    "BOMBINHAS",
                    "PENHA",
                    "PORTO BELO",
                    "BALN. PIÇARRAS",
                    "ILHOTA",
                    "LUIZ ALVES",
                    "Unknown",
                    "Other/anonymized",
                ],
            )

        with appointment_col:
            specialty = st.selectbox(
                "Specialty",
                classifier.get("specialties", ["physiotherapy"]),
            )
            appointment_date = st.date_input(
                "Appointment date", value=date.today() + timedelta(days=1)
            )
            appointment_time = st.slider(
                "Appointment hour", min_value=7, max_value=18, value=10
            )
            appointment_shift = st.selectbox(
                "Appointment shift", ["morning", "afternoon"]
            )
            sms_received = st.checkbox("SMS reminder received")

        with clinical_col:
            hypertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            alcoholism = st.checkbox("Alcoholism")
            handcap = st.checkbox("Handicap indicator")
            scholarship = st.checkbox("Scholarship benefit")
            average_temp = st.number_input(
                "Average temperature (°C)", min_value=-5.0, max_value=50.0, value=20.0
            )
            max_temp = st.number_input(
                "Maximum temperature (°C)", min_value=-5.0, max_value=55.0, value=25.0
            )

        with st.expander("Rain and prior-day conditions"):
            weather_left, weather_right = st.columns(2)
            with weather_left:
                average_rain = st.number_input(
                    "Average rainfall", min_value=0.0, value=0.0
                )
                max_rain = st.number_input(
                    "Maximum rainfall", min_value=0.0, value=0.0
                )
                rain_intensity = st.selectbox(
                    "Rain intensity", ["no_rain", "weak", "moderate", "heavy"]
                )
            with weather_right:
                heat_intensity = st.selectbox(
                    "Heat intensity",
                    ["cold", "heavy_cold", "mild", "warm", "heavy_warm"],
                    index=2,
                )
                rainy_day_before = st.checkbox("Rainy day before")
                storm_day_before = st.checkbox("Storm day before")

        submitted = st.form_submit_button(
            "Estimate no-show risk", type="primary", icon=":material/analytics:"
        )

    if submitted:
        if max_temp < average_temp:
            st.error(
                "Maximum temperature must be at least the average temperature.",
                icon=":material/error:",
            )
        else:
            row = pd.DataFrame(
                [
                    {
                        "specialty": specialty,
                        "appointment_time": appointment_time,
                        "gender": gender,
                        "disability": disability,
                        "place": place,
                        "appointment_shift": appointment_shift,
                        "age": age,
                        "under_12_years_old": int(age < 12),
                        "over_60_years_old": int(age > 60),
                        "patient_needs_companion": int(needs_companion),
                        "average_temp_day": average_temp,
                        "average_rain_day": average_rain,
                        "max_temp_day": max_temp,
                        "max_rain_day": max_rain,
                        "rainy_day_before": int(rainy_day_before),
                        "storm_day_before": int(storm_day_before),
                        "rain_intensity": rain_intensity,
                        "heat_intensity": heat_intensity,
                        "appointment_date_continuous": pd.Timestamp(appointment_date),
                        "Hipertension": int(hypertension),
                        "Diabetes": int(diabetes),
                        "Alcoholism": int(alcoholism),
                        "Handcap": int(handcap),
                        "Scholarship": int(scholarship),
                        "SMS_received": int(sms_received),
                    }
                ]
            )
            probability = float(predict_no_show(classifier, row)[0])
            threshold = float(classifier["threshold"])
            label = risk_label(probability, threshold)
            with st.container(border=True):
                st.subheader(f"{label} no-show risk")
                st.metric("Model score", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
                if label == "High":
                    st.write(
                        "Suggested action: prioritize a confirmation message or call and "
                        "offer a simple rescheduling option."
                    )
                elif label == "Moderate":
                    st.write("Suggested action: send the standard reminder sequence.")
                else:
                    st.write("Suggested action: keep the normal appointment workflow.")
                st.caption(
                    f"The operating threshold selected on validation data is {threshold:.1%}."
                )

with forecast_tab:
    st.header("Daily demand forecast")
    st.write(
        "Forecast scheduled appointment counts from historical specialty-level patterns. "
        "Longer horizons are recursively generated and carry more uncertainty."
    )
    with st.form("forecast_form", border=True):
        form_left, form_right = st.columns([1, 2])
        with form_left:
            horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
        with form_right:
            selected_specialties = st.multiselect(
                "Specialties",
                forecaster["specialties"],
                default=forecaster["specialties"],
            )
        forecast_submitted = st.form_submit_button(
            "Create forecast", type="primary", icon=":material/calendar_month:"
        )

    if forecast_submitted:
        if not selected_specialties:
            st.warning("Select at least one specialty.", icon=":material/warning:")
        else:
            with st.spinner("Generating the recursive forecast..."):
                forecast = forecast_demand(forecaster, horizon)
            detail = forecast[forecast["specialty"].isin(selected_specialties)].copy()
            combined = detail.groupby("date", as_index=False)["forecast"].sum()

            with st.container(horizontal=True):
                st.metric(
                    "Forecast appointments",
                    f"{combined['forecast'].sum():,.0f}",
                    border=True,
                )
                st.metric(
                    "Average per day",
                    f"{combined['forecast'].mean():,.1f}",
                    border=True,
                )
                peak = combined.loc[combined["forecast"].idxmax()]
                st.metric(
                    "Peak day",
                    pd.Timestamp(peak["date"]).strftime("%d %b %Y"),
                    f"{peak['forecast']:.0f} appointments",
                    border=True,
                )

            chart = (
                alt.Chart(detail)
                .mark_line(point=False)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("forecast:Q", title="Forecast appointments"),
                    color=alt.Color("specialty:N", title="Specialty"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("specialty:N", title="Specialty"),
                        alt.Tooltip("forecast:Q", title="Forecast", format=".1f"),
                    ],
                )
                .properties(height=380)
                .interactive()
            )
            st.altair_chart(chart)

            daily_table = detail.pivot(
                index="date", columns="specialty", values="forecast"
            ).reset_index()
            daily_table["Selected total"] = daily_table.drop(columns="date").sum(axis=1)
            st.dataframe(
                daily_table,
                hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="DD MMM YYYY")
                },
            )
            st.download_button(
                "Download forecast CSV",
                data=daily_table.to_csv(index=False).encode("utf-8"),
                file_name="medical_appointment_demand_forecast.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            st.caption(
                "Backtest method: rolling one-day-ahead prediction using actual prior-day "
                "history. Future multi-day forecasts use prior predictions as lag inputs."
            )
