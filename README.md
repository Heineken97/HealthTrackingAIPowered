# HealthTrackingAIPowered
Solution Joseph Jimenez:  Personal Health Record Manager: A Secure, Smart, and User-Centric Health Portal  Track: Deployment of Small Models

Clinical AI System for MIT Presentation

1. Time Series Trend Model (model_TimeSeriesTrend.py)
This module trains and manages Prophet-based forecasting models for six critical blood biomarkers (hemoglobin, WBC, glucose, creatinine, potassium, and sodium). It processes MIMIC-III lab data, aggregates daily averages, and generates 30-day forecasts with confidence intervals. Key features:

Automatically detects anomalies (values outside 95% prediction intervals).
Tracks trend direction (increasing, decreasing, or stable).
Saves trained models in /models/time_series/ as .pkl files.
Input: CSV with subject_id, charttime (timestamps), valuenum (measurement), and label (test type).
Output: JSON with forecasts, anomalies, and clinical reference ranges.

2. Clinical Report Interpreter (model_ClinicalReportInterpretation.py)
A multi-label Random Forest classifier that analyzes clinical notes and lab results to predict diagnoses. It:

Extracts symptoms from free-text notes using keyword matching (e.g., "fever" → pyrexia).
Normalizes lab values against clinical ranges (e.g., hemoglobin < 13.5 g/dL → "low").
Maps findings to 504 ICD-9 diagnosis codes with confidence scores.
Input: Clinical text + lab values (e.g., {"HEMOGLOBIN": 12.5, "WBC": 13.2}).
Output: Ranked diagnoses (e.g., ["ANEMIA", "INFECTION"]), lab alerts, and detected symptoms.

Generate de models and you can mesure it.

3. Prediction Scripts (predict_*.py)
predict_TimeSeriesTrend.py: Loads trained models, evaluates test data, and outputs forecasts with MAE/RMSE metrics.
predict_ClinicalReportInterpretation.py: Generates real-time predictions and evaluates model performance (precision, recall, F1-score). Both scripts save metrics to /metrics/ in JSON format.

Then you can use it on your app.

