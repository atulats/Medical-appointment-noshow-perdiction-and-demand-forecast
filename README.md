# Medical appointment no-show prediction and demand forecasting

This project implements the two systems required by the supplied brief:

- A binary classifier that returns a no-show risk score and an operating decision using a validation-tuned threshold.
- A specialty-level daily demand model with coherent overall totals and recursive future forecasting.
- A Streamlit app with a no-show input form and a demand-planning interface.

## Important findings from the supplied data

- The CSV contains 109,593 rows, 26 columns, 498 continuous dates, and a 31.78% no-show rate.
- Age, specialty, disability, place, and weather fields contain missing values.
- `place` has more than 26,000 distinct values. Many resemble synthetic names rather than the stated 13 municipalities. The pipeline keeps recognized municipalities and groups the rest as `other_or_anonymized` to prevent a huge, unstable feature matrix.
- Weather values are not constant within the same appointment date. They are retained for appointment-level classification, but excluded from the daily forecast because they do not form a coherent daily series and future weather is not supplied.
- Daily appointment counts are highly volatile. The training script reports achieved metrics and does not claim the brief's target metrics unless the held-out results meet them.

## Method

### No-show classification

The code uses chronological train, validation, and test periods so future appointments do not influence earlier predictions. It compares:

1. Class-weighted logistic regression
2. Class-weighted histogram gradient boosting
3. Class-weighted random forest

The decision threshold is selected only on the validation period to maximize F1. The final report includes F1, ROC-AUC, precision, recall, average precision, a confusion matrix, test predictions, and permutation feature importance.

### Demand forecasting

Daily counts are created for every date and specialty, including zero-demand combinations. The feature set includes calendar effects, trend, lags at 1, 2, 7, 14, and 28 days, and shifted rolling statistics. The code compares:

1. Poisson regression
2. Poisson histogram gradient boosting
3. Random forest regression
4. A 7-day seasonal-naive benchmark

The final forecast model is selected on a chronological holdout. Reported backtest metrics are rolling one-day-ahead metrics using actual lag history. Multi-day future forecasts are recursive, so uncertainty increases with the horizon.

## Verified results on the supplied CSV

The full workflow was executed against `Medical_appointment_data.csv`.

| Task | Selected model | Held-out result | Brief target | Status |
| --- | --- | --- | --- | --- |
| No-show classification | Histogram gradient boosting | F1 0.634, ROC-AUC 0.790, precision 0.516, recall 0.824 | F1 > 0.70 and ROC-AUC > 0.75 | ROC-AUC met; F1 not met |
| Daily demand forecasting | Histogram gradient boosting | MAE 145.7, RMSE 262.5, MAPE 2,638.5%, R² 0.137 | MAPE < 20% and R² > 0.65 | Not met |

The classification test period is 2 February 2021 through 12 May 2021. The forecast test period is 8 February 2021 through 12 May 2021. `place_group` is the strongest classification feature, but this result is not safe to interpret as a genuine geographic effect: recognized municipalities have roughly 6%-14% no-show rates, while rows with synthetic-looking place names have about a 50% no-show rate. This indicates a data-generation or anonymization artifact.

The forecast target is not attainable with the current daily series. Counts range from 1 to 1,512 appointments per day and collapse to mostly single digits near the end of the dataset. The app remains useful as a reproducible prototype, but operational forecasting requires a corrected appointment date series and a stable data-generation process.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
python train_models.py
```

Optional arguments:

```powershell
python train_models.py --data Medical_appointment_data.csv --output-dir . --intervention-effectiveness 0.15
```

Outputs are written to:

- `models/no_show_classifier.joblib`
- `models/demand_forecaster.joblib`
- `reports/` for model comparisons, metrics, predictions, feature importance, data-quality summaries, charts, and the intervention scenario

The business-impact estimate is explicitly a scenario. Replace the default 15% intervention-effectiveness assumption with evidence from a reminder pilot before using it for financial planning.

## Run the application

```powershell
streamlit run streamlit_app.py
```

The app validates inputs, caches model loading, presents a no-show action recommendation, forecasts selected specialties, and allows forecast download as CSV.

## Production considerations

- Retrain on a schedule and monitor data drift, calibration, subgroup performance, threshold performance, and forecast error.
- Validate the municipality field upstream; do not treat anonymized/synthetic locations as real geography.
- Replace the row-level weather fields with a trusted daily weather source before using weather in demand planning.
- Use the risk score only to allocate supportive outreach. Do not use it to reduce access to care.
- Track reminder outreach and outcomes so intervention effectiveness can be estimated rather than assumed.
