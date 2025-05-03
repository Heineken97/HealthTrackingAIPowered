"""
MIT Time Series Trend Prediction - Corrected Version
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

MODEL_DIR = 'models/time_series'
METRICS_FILE = 'metrics/ts_metrics.json'
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

class TimeSeriesPredictor:
    def __init__(self):
        self.models = self._load_models()
        self.metrics = {
            'last_evaluation': datetime.now().strftime("%Y-%m-%d"),
            'models': {}
        }
    
    def _load_models(self):
        """Load trained time series models"""
        models = {}
        for model_file in os.listdir(MODEL_DIR):
            if model_file.startswith('ts_') and model_file.endswith('.pkl'):
                component = model_file.split('_')[1].upper()
                models[component] = joblib.load(os.path.join(MODEL_DIR, model_file))
        return models
    
    def predict(self, patient_data, days_forward=30):
        """Generate predictions for patient lab trends"""
        results = {}
        
        for component, model_data in self.models.items():
            try:
                # Filter and prepare data
                history = patient_data[patient_data['label'].str.contains(component, case=False)]
                if len(history) < 10:
                    continue
                    
                ts_data = history.groupby(pd.Grouper(key='charttime', freq='D'))['valuenum'].mean().reset_index()
                ts_data = ts_data.rename(columns={'charttime': 'ds', 'valuenum': 'y'}).dropna()
                
                if len(ts_data) < 2:
                    continue
                
                # Make forecast
                future = model_data['model'].make_future_dataframe(periods=days_forward, include_history=True)
                forecast = model_data['model'].predict(future)
                
                # Calculate metrics
                merged = pd.merge(ts_data, forecast, on='ds', how='inner')
                metrics = None
                if len(merged) > 1:
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    metrics = {
                        'mae': float(mean_absolute_error(y_true, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    }
                
                # Prepare output
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
                
                results[component] = {
                    'forecast': forecast_data.to_dict('records'),
                    'trend': self._determine_trend(forecast),
                    'anomalies': self._detect_anomalies(forecast),
                    'reference_range': [float(x) for x in model_data['reference_range']],
                    'metrics': metrics
                }
                
            except Exception as e:
                print(f"Error processing {component}: {str(e)}")
        
        return results
    
    def evaluate_models(self, test_data):
        """Evaluate model performance on test data"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'overall_metrics': {}
        }
        
        for component, model_data in self.models.items():
            try:
                test_cases = test_data[test_data['label'].str.contains(component, case=False)]
                if len(test_cases) < 10:
                    continue
                    
                ts_test = test_cases.groupby(pd.Grouper(key='charttime', freq='D'))['valuenum'].mean().reset_index()
                ts_test = ts_test.rename(columns={'charttime': 'ds', 'valuenum': 'y'}).dropna()
                
                if len(ts_test) < 2:
                    continue
                
                future = model_data['model'].make_future_dataframe(periods=0, include_history=True)
                forecast = model_data['model'].predict(future)
                
                merged = pd.merge(ts_test, forecast, on='ds', how='inner')
                if len(merged) < 2:
                    continue
                
                y_true = merged['y'].values
                y_pred = merged['yhat'].values
                
                evaluation['models'][component] = {
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'error_distribution': {
                        'min': float(np.min(y_true - y_pred)),
                        'max': float(np.max(y_true - y_pred)),
                        'mean': float(np.mean(y_true - y_pred)),
                        'std': float(np.std(y_true - y_pred))
                    },
                    'sample_size': len(y_true),
                    'reference_range': [float(x) for x in model_data['reference_range']]
                }
                
            except Exception as e:
                print(f"Error evaluating {component}: {str(e)}")
        
        # Calculate overall metrics if any models were evaluated
        if evaluation['models']:
            all_mae = [m['mae'] for m in evaluation['models'].values()]
            all_rmse = [m['rmse'] for m in evaluation['models'].values()]
            
            evaluation['overall_metrics'] = {
                'mean_mae': float(np.mean(all_mae)),
                'mean_rmse': float(np.mean(all_rmse)),
                'best_model': min(evaluation['models'].items(), key=lambda x: x[1]['mae'])[0],
                'worst_model': max(evaluation['models'].items(), key=lambda x: x[1]['mae'])[0]
            }
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        return evaluation
    
    def _determine_trend(self, forecast):
        """Determine overall trend from forecast"""
        trend_coef = forecast['trend'].diff().mean()
        if trend_coef > 0.1:
            return "increasing"
        elif trend_coef < -0.1:
            return "decreasing"
        return "stable"
    
    def _detect_anomalies(self, forecast):
        """Detect anomaly points in forecast"""
        anomalies = forecast[
            (forecast['yhat_lower'] > forecast['yhat']) | 
            (forecast['yhat_upper'] < forecast['yhat'])
        ]
        return anomalies['ds'].dt.strftime('%Y-%m-%d').tolist()

if __name__ == "__main__":
    try:
        predictor = TimeSeriesPredictor()
        
        # Generate realistic test data aligned with model training periods
        test_data = []
        for component, model_data in predictor.models.items():
            if 'training_data' in model_data:
                dates = model_data['training_data']['ds']
                values = np.random.normal(
                    loc=np.mean(model_data['reference_range']),
                    scale=(model_data['reference_range'][1] - model_data['reference_range'][0])/6,
                    size=len(dates)
                )
                
                test_data.append(pd.DataFrame({
                    'charttime': dates,
                    'valuenum': values,
                    'label': [component] * len(dates)
                }))
        
        if not test_data:
            # Fallback if no model data available
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            dates = pd.date_range(start=start_date, end=end_date)
            test_data = pd.DataFrame({
                'charttime': dates,
                'valuenum': np.random.normal(loc=14, scale=1, size=len(dates)),
                'label': ['HEMOGLOBIN'] * len(dates)
            })
        else:
            test_data = pd.concat(test_data)
        
        # Make predictions
        predictions = predictor.predict(test_data)        
        # Evaluate models
        evaluation = predictor.evaluate_models(test_data)
        print("\nEvaluation Metrics:", json.dumps(evaluation, indent=2))
        
    except Exception as e:
        print(f"Runtime error: {str(e)}")