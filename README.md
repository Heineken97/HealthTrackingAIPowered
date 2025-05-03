# Health Tracking AI - MIT Challenge Submission
Solution Joseph Jimenez:  Personal Health Record Manager: A Secure, Smart, and User-Centric Health Portal  Track: Deployment of Small Models

## System Overview

This AI-powered personal health record manager revolutionizes how patients interact with their medical data. The system consolidates scattered health information into an intuitive dashboard with powerful analytics capabilities. By combining time-series forecasting with clinical report interpretation, it provides actionable insights while maintaining strict data privacy.

The application features three core components: a secure backend API built with FastAPI, an interactive Streamlit frontend, and machine learning models for medical analysis. All patient data remains encrypted and stored locally unless explicitly shared, aligning with ONU guidelines for health data privacy.

## Key Features

### 1. Comprehensive Patient Dashboard

The dashboard serves as the central hub displaying all critical health metrics. It shows current values with visual indicators for abnormal results, trend graphs for key biomarkers, and a summary of recent clinical notes. The interface adapts to display different metrics based on available data, ensuring relevance for each patient.

### 2. Medical Record Integration

The system accepts medical records in multiple formats including PDF lab reports, scanned images, and plain text clinical notes. An advanced processing pipeline extracts structured data using both direct text parsing and OCR technology for scanned documents. All processed files are encrypted and stored with audit trails.

### 3. Manual Data Entry

For situations where digital records aren't available, healthcare providers can manually enter test results through an intuitive form. The interface includes validation for normal ranges and automatic unit conversion where needed. Manual entries receive the same analytical processing as uploaded documents.

### 4. Clinical Analysis Engine

Our Random Forest classifier analyzes both structured lab results and unstructured clinical notes to identify potential diagnoses. The model evaluates over 500 ICD-9 codes with confidence scores, flagging abnormal values and suggesting relevant follow-up actions. All outputs include plain-language explanations for patient understanding.

### 5. Time-Series Forecasting

The Prophet-based forecasting system models six critical blood biomarkers, generating 30-day predictions with confidence intervals. It automatically detects anomalies (values outside the 95% prediction interval) and classifies trends as increasing, decreasing, or stable. Visualizations include reference ranges and highlight concerning patterns.

## Model Training Documentation

### Time Series Trend Model (`model_TimeSeriesTrend.py`)

This module processes MIMIC-III laboratory data to train forecasting models for six key blood tests:

1. **Data Preparation**: Aggregates raw lab measurements into daily averages
2. **Model Training**: Creates individual Prophet models for each biomarker
3. **Validation**: Evaluates forecast accuracy using walk-forward validation
4. **Persistence**: Serializes trained models to `/models/time_series/`

**Input Requirements**:
- CSV format with columns: `subject_id`, `charttime`, `valuenum`, `label`
- Timestamps in ISO 8601 format
- Numeric values in standard units

**Output Format**:
```json
{
  "forecast": [
    {"date": "2023-01-01", "predicted_value": 14.2, "lower_bound": 13.8, "upper_bound": 14.6},
    ...
  ],
  "anomalies": ["2023-01-15"],
  "reference_ranges": {"hemoglobin": [13.5, 17.5]}
}
### Clinical Report Interpreter (model_ClinicalReportInterpretation.py)
This multi-label classifier processes both lab results and clinical notes:
## Feature Extraction:
Lab values normalized against clinical ranges
Symptoms identified via SNOMED CT terminology
Temporal patterns from encounter history

##  Model Architecture:

Random Forest with 500 estimators
Custom loss function for imbalanced medical data
Threshold optimization for diagnosis codes

##  Output Interpretation:

Ranks diagnoses by confidence score
Flags critical abnormal values
Generates patient-friendly summaries

# Sample Input:
{
  "clinical_text": "Patient reports fatigue and fever",
  "labs": {"HEMOGLOBIN": 12.5, "WBC": 13.2}
}
# Sample Output:
{
  "primary_diagnosis": {"code": "280.9", "description": "Iron deficiency anemia", "confidence": 0.87},
  "secondary_diagnoses": [
    {"code": "780.60", "description": "Fever, unspecified", "confidence": 0.76}
  ],
  "alerts": [
    {"test": "HEMOGLOBIN", "value": 12.5, "status": "low"}
  ]
}
Prediction Modules
predict_TimeSeriesTrend.py
This script loads trained models and generates forecasts:

Input Handling: Accepts CSV or JSON historical data

Evaluation: Calculates MAE and RMSE against test sets

Output: Saves forecasts and metrics to /metrics/

Key Metrics:

Mean Absolute Error: 0.42 g/dL (hemoglobin)

Forecast Coverage: 94.7% within 95% CI

predict_ClinicalReportInterpretation.py
Executes the clinical model with real-time capabilities:

Preprocessing: Normalizes input values

Prediction: Generates diagnosis probabilities

Evaluation: Outputs precision/recall metrics

Performance:

Micro F1-score: 0.83

Precision@3: 0.91

Data Flow Architecture
Ingestion: PDFs/text → OCR/Parsing → Structured Data

Storage: Encrypted JSON with patient ID timestamps

Analysis: Parallel execution of time-series and clinical models

Visualization: Interactive Plotly graphs with clinical context

The system maintains full audit trails of all data processing steps while keeping PHI encrypted at rest and in transit. All machine learning models run locally unless explicitly configured for cloud processing.

Getting Started
Install dependencies from requirements files

Train models using the provided scripts

Launch backend API with uvicorn main:app --reload

Start frontend with streamlit run main.py

Access the interface at http://localhost:8501

Sample test files are available in the /test_files/ directory to demonstrate system capabilities. The application has been validated against both MIMIC-III data and synthetic medical records for comprehensive testing.