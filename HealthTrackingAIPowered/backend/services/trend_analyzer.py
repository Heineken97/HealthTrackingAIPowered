"""
Time Series Trend Analysis Module
MIT Challenge Submission

Implements Facebook Prophet for forecasting medical trends
with anomaly detection and visualization support.
"""

import pandas as pd
from prophet import Prophet
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    def __init__(self):
        """Initialize trend analyzer with reference ranges"""
        self.reference_ranges = {
            "hemoglobin": (13.5, 17.5),  # g/dL
            "glucose": (70, 100),        # mg/dL
            "wbc": (4.5, 11.0),          # 10^3 cells/μL
            "creatinine": (0.7, 1.3)     # mg/dL
        }
        self.models = {}

    def load_models(self):
        """Load pre-trained models for each test type"""
        try:
            # In production, load actual serialized models
            logger.info("Trend analysis models initialized")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def analyze(self, test_name: str, history: List[Dict]) -> Dict:
        """
        Analyze time-series medical data
        
        Args:
            test_name: Laboratory test name
            history: List of historical measurements
            
        Returns:
            Dictionary containing:
            - forecast: Predicted values with confidence intervals
            - trend: Overall trend direction
            - anomalies: Dates with abnormal values
        """
        try:
            # Convert history to DataFrame
            df = pd.DataFrame([{
                'ds': pd.to_datetime(h['date']),
                'y': h['value']
            } for h in history])
            
            # Train and forecast
            model = Prophet()
            model.fit(df)
            
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(df, test_name)
            
            # Determine overall trend
            trend = self._determine_trend(df['y'])
            
            return {
                "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(),
                "trend": trend,
                "anomalies": anomalies
            }
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            raise

    def _detect_anomalies(self, df: pd.DataFrame, test_name: str) -> List[str]:
        """Identify abnormal values based on reference ranges"""
        if test_name not in self.reference_ranges:
            return []
            
        ref_min, ref_max = self.reference_ranges[test_name]
        anomalies = df[(df['y'] < ref_min) | (df['y'] > ref_max)]
        return anomalies['ds'].dt.strftime('%Y-%m-%d').tolist()

    def _determine_trend(self, values: pd.Series) -> str:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return "insufficient_data"
            
        slope = (values.iloc[-1] - values.iloc[0]) / len(values)
        
        if abs(slope) < 0.1:
            return "stable"
        return "increasing" if slope > 0 else "decreasing"