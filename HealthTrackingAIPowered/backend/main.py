"""
Health Tracking AI - Backend API
MIT Challenge Submission

Features:
1. Secure medical record processing
2. AI-powered clinical analysis
3. Time-series trend forecasting
4. Encrypted data storage
"""

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict
import os
import json
import logging

# Internal imports
from utils.file_processor import process_medical_file
from utils.data_security import encrypt_data, decrypt_data
from services.clinical_analyzer import ClinicalAnalyzer
from services.trend_analyzer import TrendAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Health Tracking AI API",
    description="MIT Challenge Submission - AI-powered personal health record system",
    version="1.0",
    docs_url="/api/docs",
    redoc_url=None,
    contact={
        "name": "Joseph David Jimenez Zuniga",
        "url": "https://github.com/Heineken97",
        "email": "josephdjz@hotmail.com"
    }
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class PatientDemographics(BaseModel):
    """Patient demographic information model"""
    patient_id: str
    age: int
    weight: float  # in kg
    height: float  # in cm
    gender: Optional[str] = "unspecified"
    ethnicity: Optional[str] = "unspecified"

class LabValue(BaseModel):
    """Laboratory test result model"""
    test_name: str
    value: float
    unit: str
    date: datetime


# Initialize services
clinical_analyzer = ClinicalAnalyzer()
trend_analyzer = TrendAnalyzer()

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    try:
        clinical_analyzer.load_models()
        trend_analyzer.load_models()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise

@app.post("/enter_manual_data", tags=["Medical Records"])
async def enter_manual_data(
    patient: PatientDemographics,
    lab_values: Dict[str, float],
    clinical_note: Optional[str] = None
):
    """
    Enter medical data manually
    
    Args:
        patient: Patient demographic information
        lab_values: Laboratory test results
        clinical_note: Optional clinical notes
        
    Returns:
        Processing status with patient ID
    """
    try:
        record = {
            **patient.dict(),
            "lab_values": lab_values,
            "clinical_note": clinical_note,
            "date": datetime.now().isoformat(),
            "source": "manual_entry"
        }
        
        os.makedirs("data/patients", exist_ok=True)
        encrypted_data = encrypt_data(record)
        
        output_filename = f"data/patients/{patient.patient_id}_{datetime.now().timestamp()}.enc"
        with open(output_filename, "wb") as f:
            f.write(encrypted_data)
            
        logger.info(f"Manual data saved for patient {patient.patient_id}")
        return {"status": "success", "patient_id": patient.patient_id}
    except Exception as e:
        logger.error(f"Manual entry failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Manual data entry error")

@app.post("/upload", tags=["Medical Records"])
async def upload_medical_record(
    file: UploadFile,
    patient: PatientDemographics,
    background_tasks: BackgroundTasks
):
    """
    Upload and process medical records
    
    Args:
        file: Medical record file (PDF, image, or text)
        patient: Patient demographic information
        background_tasks: FastAPI background tasks
        
    Returns:
        Processing status with patient ID
    """
    try:
        file_bytes = await file.read()
        
        background_tasks.add_task(
            process_medical_background,
            file_bytes,
            patient.dict(),
            file.filename
        )
        
        return {"status": "processing", "patient_id": patient.patient_id}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="File processing error")

def process_medical_background(file_bytes: bytes, patient: dict, filename: str):
    """Background task for medical file processing"""
    try:
        extracted_data = process_medical_file(file_bytes)
        
        record = {
            **patient,
            **extracted_data,
            "original_filename": filename,
            "processed_at": datetime.now().isoformat()
        }
        
        os.makedirs("data/patients", exist_ok=True)
        encrypted_data = encrypt_data(record)
        
        output_filename = f"data/patients/{patient['patient_id']}_{datetime.now().timestamp()}.enc"
        with open(output_filename, "wb") as f:
            f.write(encrypted_data)
            
        logger.info(f"Successfully processed file for patient {patient['patient_id']}")
    except Exception as e:
        logger.error(f"Background processing failed: {str(e)}")

@app.post("/analyze/clinical", tags=["AI Analysis"])
async def analyze_clinical_data(
    clinical_text: str,
    lab_values: Dict[str, float]
):
    """
    Perform clinical analysis on medical data
    
    Args:
        clinical_text: Doctor's notes or medical observations
        lab_values: Laboratory test results
        
    Returns:
        Analysis results including diagnoses and alerts
    """
    try:
        results = clinical_analyzer.analyze(clinical_text, lab_values)
        return {
            "diagnoses": results["diagnoses"],
            "alerts": results["alerts"],
            "summary": results["summary"]
        }
    except Exception as e:
        logger.error(f"Clinical analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis error")

@app.post("/analyze/trends", tags=["AI Analysis"])
async def analyze_trend_data(
    test_name: str,
    history: List[LabValue]
):
    """
    Analyze time-series medical data
    
    Args:
        test_name: Name of the laboratory test
        history: Historical test results
        
    Returns:
        Trend analysis with forecasts and anomalies
    """
    try:
        results = trend_analyzer.analyze(test_name, history)
        return {
            "forecast": results["forecast"],
            "trend": results["trend"],
            "anomalies": results["anomalies"]
        }
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Trend analysis error")

@app.get("/records/{patient_id}", tags=["Medical Records"])
async def get_patient_records(patient_id: str):
    """
    Retrieve patient's medical records
    
    Args:
        patient_id: Unique patient identifier
        
    Returns:
        List of encrypted patient records
    """
    try:
        records = []
        for filename in os.listdir("data/patients"):
            if filename.startswith(patient_id):
                with open(f"data/patients/{filename}", "rb") as f:
                    encrypted = f.read()
                    records.append(decrypt_data(encrypted))
        
        return {"patient_id": patient_id, "records": records}
    except Exception as e:
        logger.error(f"Record retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Data retrieval error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)