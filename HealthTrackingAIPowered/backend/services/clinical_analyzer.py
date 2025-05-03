"""
Clinical Analysis Module
MIT Challenge Submission

Implements medical diagnosis prediction and report interpretation
using trained machine learning models.
"""

import joblib
import pandas as pd
from typing import Dict, List
import os
import logging

logger = logging.getLogger(__name__)

class ClinicalAnalyzer:
    def __init__(self):
        """Initialize clinical analyzer with reference ranges"""
        self.model = None
        self.reference_ranges = {
            "hemoglobin": (13.5, 17.5),
            "glucose": (70, 100),
            "wbc": (4.5, 11.0),
            "creatinine": (0.7, 1.3)
        }
        self.diagnosis_labels = [
            "Normal", 
            "Type 2 Diabetes", 
            "Anemia",
            "Hypertension",
            "Kidney Disease"
        ]

    def load_models(self):
        """Load trained clinical models from disk"""
        try:
            # In production, load actual trained models
            # Example: self.model = joblib.load('models/clinical_model.pkl')
            logger.info("Clinical models initialized")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def analyze(self, clinical_text: str, lab_values: Dict[str, float]) -> Dict:
        """
        Perform clinical analysis on medical data
        
        Args:
            clinical_text: Doctor's notes or observations
            lab_values: Laboratory test results
            
        Returns:
            Dictionary containing:
            - diagnoses: List of potential diagnoses
            - alerts: Abnormal lab values
            - summary: Plain English explanation
        """
        try:
            # Validate and normalize inputs
            validated_data = self._validate_inputs(lab_values)
            
            # Extract features from text
            text_features = self._extract_text_features(clinical_text)
            
            # Combine features
            features = {**validated_data, **text_features}
            
            # Make prediction (mock for demonstration)
            diagnosis = self._predict_diagnosis(features)
            alerts = self._check_abnormal_values(validated_data)
            summary = self._generate_summary(diagnosis, alerts)
            
            return {
                "diagnoses": diagnosis,
                "alerts": alerts,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Clinical analysis failed: {str(e)}")
            raise

    def _validate_inputs(self, lab_values: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize lab values"""
        validated = {}
        for test, value in lab_values.items():
            if test in self.reference_ranges:
                validated[test] = float(value)
        return validated

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract clinical features from text"""
        # In production, use NLP pipeline
        return {
            "has_fever": 1.0 if "fever" in text.lower() else 0.0,
            "has_fatigue": 1.0 if "fatigue" in text.lower() else 0.0
        }

    def _predict_diagnosis(self, features: Dict) -> List[Dict]:
        """Predict potential diagnoses"""
        # Mock prediction for demonstration
        return [{
            "condition": "Anemia",
            "confidence": 0.85
        }, {
            "condition": "Type 2 Diabetes",
            "confidence": 0.45
        }]

    def _check_abnormal_values(self, lab_values: Dict[str, float]) -> List[Dict]:
        """Identify abnormal lab values"""
        alerts = []
        for test, value in lab_values.items():
            if test in self.reference_ranges:
                ref_min, ref_max = self.reference_ranges[test]
                if value < ref_min or value > ref_max:
                    alerts.append({
                        "test": test,
                        "value": value,
                        "status": "low" if value < ref_min else "high",
                        "reference_range": [ref_min, ref_max]
                    })
        return alerts

    def _generate_summary(self, diagnosis: List[Dict], alerts: List[Dict]) -> str:
        """Generate plain English summary"""
        summary = "Clinical Analysis Summary:\n\n"
        
        if diagnosis:
            summary += "Potential Diagnoses:\n"
            for dx in diagnosis:
                summary += f"- {dx['condition']} (confidence: {dx['confidence']:.0%})\n"
        
        if alerts:
            summary += "\nAlerts:\n"
            for alert in alerts:
                summary += (
                    f"- Abnormal {alert['test']}: {alert['value']} "
                    f"(reference: {alert['reference_range'][0]}-{alert['reference_range'][1]})\n"
                )
        
        return summary