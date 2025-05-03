"""
MIT Clinical Report Interpretation System
----------------------------------------

This module implements a machine learning pipeline for interpreting clinical reports
and laboratory results to suggest potential diagnoses using Random Forest classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
from datetime import datetime
import json
import re

# Configuration
DATA_DIR = 'datasets/mimic-iii'
MODEL_DIR = 'models/clinical'
os.makedirs(MODEL_DIR, exist_ok=True)

class ClinicalInterpreter:
    def __init__(self):
        """Initialize interpreter with clinical reference ranges and symptom keywords."""
        # Normal laboratory value ranges
        self.lab_ref_ranges = {
            'HEMOGLOBIN': (13.5, 17.5),  # g/dL
            'WBC': (4.5, 11.0),          # 10^3 cells/μL  
            'GLUCOSE': (70, 100),        # mg/dL
            'CREATININE': (0.7, 1.3)     # mg/dL
        }
        
        # Symptom keywords for text extraction
        self.symptom_keywords = {
            'fatigue': ['fatigue', 'tired', 'exhaust'],
            'fever': ['fever', 'pyrexia', 'febrile'],
            'pain': ['pain', 'ache', 'discomfort'],
            'nausea': ['nausea', 'vomit', 'emesis'],
            'shortness_of_breath': ['shortness of breath', 'dyspnea']
        }
    
    def load_data(self):
        """
        Load MIMIC-III clinical data from CSV files.
        
        Returns:
            bool: True if all data loaded successfully
        """
        print("Loading MIMIC-III clinical data...")
        try:
            # Load diagnoses data
            diagnoses = pd.read_csv(
                f'{DATA_DIR}/DIAGNOSES_ICD.csv',
                usecols=['subject_id', 'icd9_code'],
                dtype={'subject_id': 'int32', 'icd9_code': 'str'}
            )
            
            icd_desc = pd.read_csv(
                f'{DATA_DIR}/D_ICD_DIAGNOSES.csv',
                usecols=['icd9_code', 'long_title'],
                dtype={'icd9_code': 'str', 'long_title': 'str'}
            )
            self.diagnoses = diagnoses.merge(icd_desc, on='icd9_code')
            
            # Load clinical notes
            self.notes = pd.read_csv(
                f'{DATA_DIR}/NOTEEVENTS.csv',
                usecols=['subject_id', 'text'],
                dtype={'subject_id': 'int32', 'text': 'str'}
            )
            
            # Load laboratory results
            self.labs = pd.read_csv(
                f'{DATA_DIR}/LABEVENTS.csv',
                usecols=['subject_id', 'itemid', 'valuenum'],
                dtype={'subject_id': 'int32', 'itemid': 'int32', 'valuenum': 'float32'}
            )
            
            # Load lab item descriptions
            lab_items = pd.read_csv(
                f'{DATA_DIR}/D_LABITEMS.csv',
                usecols=['itemid', 'label'],
                dtype={'itemid': 'int32', 'label': 'str'}
            )
            self.labs = self.labs.merge(lab_items, on='itemid')
            
            print("Data loading completed successfully")
            print(f"- Diagnoses: {len(self.diagnoses):,} records")
            print(f"- Clinical notes: {len(self.notes):,} records") 
            print(f"- Lab results: {len(self.labs):,} records")
            
            return True
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            return False
    
    def train_model(self, sample_size=500):
        """
        Train the clinical interpretation model.
        
        Args:
            sample_size (int): Number of patient cases to use for training
            
        Returns:
            bool: True if training succeeded
        """
        if not self.load_data():
            return False
            
        print("\nPreparing training data...")
        try:
            # Identify patients with both notes and diagnoses
            note_patients = set(self.notes['subject_id'].unique())
            dx_patients = set(self.diagnoses['subject_id'].unique())
            valid_patients = list(note_patients.intersection(dx_patients))
            
            if not valid_patients:
                print("Warning: No patients with both notes and diagnoses. Using all available data.")
                valid_patients = list(note_patients.union(dx_patients))
                if not valid_patients:
                    raise ValueError("No patient data available")
            
            # Sample representative patient cases
            sample_patients = np.random.choice(
                valid_patients,
                min(sample_size, len(valid_patients)),
                replace=False
            )
            
            # Prepare features and labels
            X_features = []
            y_labels = []
            
            for patient_id in sample_patients:
                # Process diagnoses (labels)
                dx = self.diagnoses[self.diagnoses['subject_id'] == patient_id]['long_title'].tolist()
                if not dx:
                    continue
                
                # Normalize diagnoses to broader categories
                normalized_dx = []
                for diagnosis in dx:
                    diagnosis_lower = diagnosis.lower()
                    if any(s in diagnosis_lower for s in ['anemia', 'hemoglobin']):
                        normalized_dx.append('ANEMIA')
                    elif any(s in diagnosis_lower for s in ['infection', 'fever', 'sepsis']):
                        normalized_dx.append('INFECTION')
                    elif any(s in diagnosis_lower for s in ['diabetes', 'glucose']):
                        normalized_dx.append('DIABETES')
                    elif any(s in diagnosis_lower for s in ['kidney', 'renal', 'creatinine']):
                        normalized_dx.append('RENAL_DISEASE')
                    else:
                        normalized_dx.append(diagnosis[:50])  # Truncate long diagnoses
                
                # Extract clinical features
                features = {
                    'symptoms': self._extract_symptoms(patient_id),
                    'labs': self._extract_labs(patient_id),
                    'age': np.random.randint(18, 90), 
                    'gender': np.random.choice(['M', 'F'])  # Placeholder
                }
                
                X_features.append(features)
                y_labels.append(list(set(normalized_dx)))  # Remove duplicates
            
            if len(X_features) < 50:
                raise ValueError(f"Insufficient training samples ({len(X_features)}) after processing")
            
            # Convert features to model input format
            X_processed = self._process_features(X_features)
            
            # Multi-label encoding for diagnoses
            self.mlb = MultiLabelBinarizer()
            y_encoded = self.mlb.fit_transform(y_labels)
            
            # Train Random Forest classifier
            print("\nTraining clinical interpretation model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            self.model.fit(X_processed, y_encoded)
            
            # Save model package
            model_data = {
                'model': self.model,
                'mlb': self.mlb,
                'symptom_keywords': self.symptom_keywords,
                'lab_ref_ranges': self.lab_ref_ranges,
                'train_date': datetime.now().strftime("%Y-%m-%d"),
                'diagnosis_classes': self.mlb.classes_.tolist()
            }
            
            joblib.dump(model_data, f'{MODEL_DIR}/clinical_model.pkl')
            
            print("\nTraining completed successfully")
            print(f"- Trained on {len(X_features)} patient cases")
            print(f"- {len(self.mlb.classes_)} diagnosis categories")
            print(f"- Model saved to {MODEL_DIR}/clinical_model.pkl")
            
            return True
            
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return False
    
    def _extract_symptoms(self, patient_id):
        """Extract symptoms from clinical notes using keyword matching."""
        notes = self.notes[self.notes['subject_id'] == patient_id]['text'].str.cat(sep=' ').lower()
        symptoms = []
        for symptom, keywords in self.symptom_keywords.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', notes) for kw in keywords):
                symptoms.append(symptom)
        return symptoms
    
    def _extract_labs(self, patient_id):
        """Extract laboratory features for a patient."""
        labs = self.labs[self.labs['subject_id'] == patient_id]
        lab_features = {}
        for test, (ref_min, ref_max) in self.lab_ref_ranges.items():
            test_values = labs[labs['label'].str.contains(test, case=False)]['valuenum']
            if not test_values.empty:
                lab_features[test] = test_values.mean()
        return lab_features
    
    def _process_features(self, X):
        """Convert feature dictionary to numerical matrix."""
        # Symptom features (one-hot encoded)
        symptom_features = np.zeros((len(X), len(self.symptom_keywords)))
        for i, x in enumerate(X):
            for j, symptom in enumerate(self.symptom_keywords):
                if symptom in x['symptoms']:
                    symptom_features[i, j] = 1
        
        # Laboratory features (normalized by reference range)
        lab_features = np.zeros((len(X), len(self.lab_ref_ranges)))
        for i, x in enumerate(X):
            for j, test in enumerate(self.lab_ref_ranges):
                if test in x['labs']:
                    ref_min, ref_max = self.lab_ref_ranges[test]
                    lab_features[i, j] = (x['labs'][test] - ref_min) / (ref_max - ref_min)
        
        # Demographic features
        demo_features = np.array([
            [x['age'] / 100.0,  # Normalized age
             1 if x['gender'] == 'M' else 0]
            for x in X
        ])
        
        return np.hstack([demo_features, symptom_features, lab_features])

if __name__ == "__main__":
    interpreter = ClinicalInterpreter()
    success = interpreter.train_model(sample_size=500)
    if not success:
        print("\nModel training failed. Please check error messages.")