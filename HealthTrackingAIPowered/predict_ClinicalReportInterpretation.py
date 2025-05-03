"""
MIT Clinical Report Interpretation Prediction - Corrected Version
"""

import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import json
import os

MODEL_DIR = 'models/clinical'
METRICS_FILE = 'metrics/clinical_metrics.json'
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

class ClinicalPredictor:
    def __init__(self):
        model_data = joblib.load(os.path.join(MODEL_DIR, 'clinical_model.pkl'))
        self.model = model_data['model']
        self.mlb = model_data['mlb']
        self.symptom_keywords = model_data['symptom_keywords']
        self.lab_ref_ranges = model_data['lab_ref_ranges']
        self.diagnosis_classes = model_data.get('diagnosis_classes', self.mlb.classes_.tolist())
        
        self.metrics = {
            'last_evaluation': datetime.now().strftime("%Y-%m-%d"),
            'performance': {}
        }
    
    def predict(self, clinical_text=None, lab_results=None, patient_id=None):
        """Generate diagnostic suggestions from clinical data"""
        try:
            # Process input features
            features = self._process_input(clinical_text, lab_results, patient_id)
            if features is None:
                return None
                
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(features)
            else:
                decision = self.model.decision_function(features)
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # Handle binary case
            if probas.ndim == 1:
                probas = probas.reshape(1, -1)
            
            # Get top predictions
            top_n = min(5, probas.shape[1])
            top_indices = np.argsort(-probas, axis=1)[:, :top_n]
            
            diagnoses = []
            for i in range(top_n):
                class_idx = top_indices[0][i]
                diagnoses.append({
                    'condition': self.diagnosis_classes[class_idx],
                    'confidence': float(probas[0][class_idx])
                })
            
            # Generate lab alerts
            alerts = []
            if lab_results:
                for test, value in lab_results.items():
                    if test in self.lab_ref_ranges:
                        ref_min, ref_max = self.lab_ref_ranges[test]
                        status = None
                        if value < ref_min:
                            status = 'low'
                        elif value > ref_max:
                            status = 'high'
                        
                        if status:
                            alerts.append({
                                'component': test,
                                'value': float(value),
                                'status': status,
                                'reference_range': [float(ref_min), float(ref_max)]
                            })
            
            return {
                'diagnosis_suggestions': diagnoses,
                'alerts': alerts,
                'symptoms_detected': self._get_detected_symptoms(features)
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def evaluate(self, test_cases):
        """Evaluate model performance"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'case_examples': []
        }
        
        try:
            X_test = []
            y_true = []
            
            # Map test diagnoses to trained classes
            for case in test_cases:
                features = self._process_input(case['text'], case['labs'])
                if features is None:
                    continue
                
                X_test.append(features[0])
                
                # Find matching diagnoses from trained classes
                matched_dx = []
                for dx in case['true_diagnoses']:
                    dx_lower = dx.lower()
                    if 'anemia' in dx_lower:
                        matched_dx.extend([c for c in self.diagnosis_classes if 'ANEMIA' in c])
                    elif 'infection' in dx_lower or 'viral' in dx_lower:
                        matched_dx.extend([c for c in self.diagnosis_classes if 'INFECTION' in c])
                    else:
                        matched_dx.extend([c for c in self.diagnosis_classes if dx.lower() in c.lower()])
                
                y_true.append(list(set(matched_dx)))  # Remove duplicates
            
            if not X_test:
                return None
                
            X_test = np.vstack(X_test)
            y_true_encoded = self.mlb.transform(y_true)
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics with zero_division parameter
            evaluation['metrics'] = {
                'precision_macro': precision_score(y_true_encoded, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true_encoded, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true_encoded, y_pred, average='macro', zero_division=0),
                'accuracy': float(np.mean(y_pred.ravel() == y_true_encoded.ravel()))
            }
            
            # Store examples
            for i in range(min(3, len(test_cases))):
                pred_labels = [self.diagnosis_classes[idx] for idx in np.where(y_pred[i])[0]]
                evaluation['case_examples'].append({
                    'input_text': test_cases[i]['text'][:200] + "...",
                    'true_diagnosis': test_cases[i]['true_diagnoses'],
                    'predicted_diagnosis': pred_labels,
                    'correct': any(any(dx.lower() in pred.lower() for pred in pred_labels) 
                                for dx in test_cases[i]['true_diagnoses'])
                })
            
            with open(METRICS_FILE, 'w') as f:
                json.dump(evaluation, f, indent=2)
            
            return evaluation
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return None
    
    def _process_input(self, clinical_text, lab_results, patient_id=None):
        """Convert input to feature vector"""
        try:
            num_symptoms = len(self.symptom_keywords)
            num_labs = len(self.lab_ref_ranges)
            features = np.zeros((1, 2 + num_symptoms + num_labs))
            
            # Demographic placeholders
            features[0, 0] = 0.5  # Normalized age
            features[0, 1] = 1    # Gender
            
            # Process symptoms
            if clinical_text:
                text = clinical_text.lower()
                for i, (_, keywords) in enumerate(self.symptom_keywords.items()):
                    if any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keywords):
                        features[0, 2+i] = 1
            
            # Process labs
            if lab_results:
                for i, test in enumerate(self.lab_ref_ranges):
                    if test in lab_results:
                        ref_min, ref_max = self.lab_ref_ranges[test]
                        normalized = (lab_results[test] - ref_min) / (ref_max - ref_min)
                        features[0, 2+num_symptoms+i] = normalized
            
            return features
            
        except Exception as e:
            print(f"Input processing error: {str(e)}")
            return None
    
    def _get_detected_symptoms(self, features):
        """Extract detected symptoms from feature vector"""
        symptom_start = 2  # After age and gender
        symptom_names = list(self.symptom_keywords.keys())
        return [symptom_names[i] 
               for i in range(len(symptom_names)) 
               if features[0, symptom_start+i] > 0]

if __name__ == "__main__":
    try:
        predictor = ClinicalPredictor()
        
        # Test case with mapped diagnoses
        test_case = {
            'text': "Patient presents with fatigue and fever for 3 days. Reports nausea after meals.",
            'labs': {'HEMOGLOBIN': 12.5, 'WBC': 13.2},
            'true_diagnoses': ['ANEMIA', 'INFECTION']  # Using trained class names
        }
        
        # Make prediction
        prediction = predictor.predict(
            clinical_text=test_case['text'],
            lab_results=test_case['labs']
        )
        print("Prediction:", json.dumps(prediction, indent=2))
        
        # Evaluation
        evaluation = predictor.evaluate([test_case])
        if evaluation:
            print("\nEvaluation Metrics:", json.dumps(evaluation, indent=2))
            
    except Exception as e:
        print(f"Runtime error: {str(e)}")