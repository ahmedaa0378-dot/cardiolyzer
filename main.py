from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Cardiac Readmission Risk API",
    description="AI-powered cardiac readmission prediction with SHAP explainability",
    version="1.1.0"
)

# Enable CORS (so Bolt.new frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Bolt.new domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================
print("Loading model artifacts...")

try:
    # Load model
    with open('readmission_model_improved.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")
    
    # Load feature config
    with open('feature_config.pkl', 'rb') as f:
        feature_config = pickle.load(f)
    print("✅ Feature config loaded")
    
    # Load SHAP explainer
    with open('shap_explainer.pkl', 'rb') as f:
        shap_explainer = pickle.load(f)
    print("✅ SHAP explainer loaded")
    
    EXPECTED_FEATURES = feature_config['feature_names']
    print(f"✅ Expected {len(EXPECTED_FEATURES)} features")
    
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")
    raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_patient_metadata(df):
    """Extract patient metadata (ID, name, age, gender, diagnosis) before preprocessing"""
    
    metadata = []
    
    # Clean column names for matching
    df_cols = df.columns.str.strip().str.lower().str.replace(' ', '_')
    col_mapping = {clean: orig for clean, orig in zip(df_cols, df.columns)}
    
    for i in range(len(df)):
        patient_meta = {
            'patient_id': None,
            'patient_name': None,
            'age': None,
            'gender': None,
            'primary_diagnosis': None
        }
        
        # Extract Patient ID
        for col_pattern in ['patient_id', 'paitient_id', 'patientid', 'patient id', 'paitient id']:
            matching_cols = [orig for clean, orig in col_mapping.items() if col_pattern in clean]
            if matching_cols:
                patient_meta['patient_id'] = str(df.iloc[i][matching_cols[0]])
                break
        
        # Extract Patient Name
        for col_pattern in ['patient_name', 'patientname', 'patient name', 'name']:
            matching_cols = [orig for clean, orig in col_mapping.items() if col_pattern in clean and 'user' not in clean]
            if matching_cols:
                val = df.iloc[i][matching_cols[0]]
                if pd.notna(val):
                    patient_meta['patient_name'] = str(val)
                break
        
        # Extract Age
        for col_pattern in ['age']:
            matching_cols = [orig for clean, orig in col_mapping.items() if clean == col_pattern or clean.endswith('_age')]
            if matching_cols:
                val = df.iloc[i][matching_cols[0]]
                if pd.notna(val):
                    try:
                        patient_meta['age'] = int(float(val))
                    except:
                        pass
                break
        
        # Extract Gender
        for col_pattern in ['gender', 'sex']:
            matching_cols = [orig for clean, orig in col_mapping.items() if col_pattern in clean]
            if matching_cols:
                val = df.iloc[i][matching_cols[0]]
                if pd.notna(val):
                    patient_meta['gender'] = str(val)
                break
        
        # Extract Primary Diagnosis
        for col_pattern in ['primary_diagnosis', 'diagnosis', 'primary diagnosis']:
            matching_cols = [orig for clean, orig in col_mapping.items() if col_pattern in clean.replace('_', ' ') or col_pattern in clean]
            if matching_cols:
                val = df.iloc[i][matching_cols[0]]
                if pd.notna(val):
                    patient_meta['primary_diagnosis'] = str(val)
                break
        
        # Fallback: generate ID if not found
        if patient_meta['patient_id'] is None:
            patient_meta['patient_id'] = str(i + 1)
        
        # Fallback: generate name if not found
        if patient_meta['patient_name'] is None:
            patient_meta['patient_name'] = f"Patient {patient_meta['patient_id']}"
        
        metadata.append(patient_meta)
    
    return metadata


def preprocess_data(df):
    """Preprocess uploaded data to match training format"""
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
    
    # Drop unnecessary columns (including metadata columns we already extracted)
    cols_to_drop = [
        'paitient_ID',
        'Patient_ID', 
        'Patient_Name',
        'Date_of_admission', 
        'Date_of_discharge',
        'High_risk_flag',
        'Risk_score_(0-10)',
        'Readmiited_within_30_days',
        'Readmitted_within_30_days'
    ]
    
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    binary_map = {'Y': 1, 'y': 1, 'N': 0, 'n': 0, 'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}
    
    for col in categorical_cols:
        unique_vals = df[col].unique()
        
        if len(unique_vals) <= 2 and any(val in ['Y', 'y', 'N', 'n', 'Yes', 'No'] for val in unique_vals if pd.notna(val)):
            df[col] = df[col].map(binary_map)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Ensure all expected features are present
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    
    # Select only the features used during training (in correct order)
    df = df[EXPECTED_FEATURES]
    
    return df


def generate_patient_explanation(patient_idx, shap_values, features):
    """Generate human-readable explanation for a patient"""
    
    patient_shap = shap_values[patient_idx]
    patient_features = features.iloc[patient_idx]
    
    shap_contrib = pd.DataFrame({
        'feature': EXPECTED_FEATURES,
        'shap_value': patient_shap,
        'feature_value': patient_features.values
    })
    
    top_positive = shap_contrib.nlargest(5, 'shap_value')
    top_negative = shap_contrib.nsmallest(5, 'shap_value')
    
    return {
        'risk_factors': [
            {
                'feature': row['feature'],
                'value': float(row['feature_value']),
                'impact': float(row['shap_value'])
            }
            for _, row in top_positive.iterrows()
        ],
        'protective_factors': [
            {
                'feature': row['feature'],
                'value': float(row['feature_value']),
                'impact': float(row['shap_value'])
            }
            for _, row in top_negative.iterrows()
        ]
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Cardiac Readmission Risk API is running",
        "version": "1.1.0",
        "model_loaded": True,
        "shap_enabled": True
    }

@app.get("/api/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": "loaded",
        "scaler": "loaded",
        "shap_explainer": "loaded",
        "expected_features": len(EXPECTED_FEATURES),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict")
async def predict_readmission(file: UploadFile = File(...)):
    """
    Predict readmission risk for uploaded patient data
    Accepts CSV or Excel files
    """
    
    try:
        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload CSV or Excel file."
            )
        
        # Read file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        else:
            from io import BytesIO
            df = pd.read_excel(BytesIO(contents))
        
        print(f"📊 Received file with {df.shape[0]} patients, {df.shape[1]} columns")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        # Validate data size
        if df.shape[0] > 10000:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum 10,000 patients allowed."
            )
        
        if df.shape[0] == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty. Please upload a file with patient data."
            )
        
        # Extract patient metadata BEFORE preprocessing
        print("📝 Extracting patient metadata...")
        patient_metadata = extract_patient_metadata(df)
        
        # Preprocess data for model
        df_processed = preprocess_data(df.copy())
        
        print(f"✅ Preprocessed to {df_processed.shape[0]} patients, {df_processed.shape[1]} features")
        
        # Scale features
        X_scaled = scaler.transform(df_processed)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate SHAP values
        print("🔍 Calculating SHAP values...")
        shap_values = shap_explainer.shap_values(X_scaled)
        
        # Generate results for each patient
        results = []
        for i in range(len(df_processed)):
            meta = patient_metadata[i]
            
            patient_result = {
                'patient_id': meta['patient_id'],
                'patient_name': meta['patient_name'],
                'age': meta['age'],
                'gender': meta['gender'],
                'primary_diagnosis': meta['primary_diagnosis'],
                'risk_prediction': int(predictions[i]),
                'risk_probability': float(prediction_proba[i]) * 100,  # Convert to percentage
                'risk_level': 'High' if predictions[i] == 1 else 'Low',
                'explanation': generate_patient_explanation(i, shap_values, df_processed)
            }
            results.append(patient_result)
        
        # Calculate summary statistics
        high_risk_count = int(predictions.sum())
        low_risk_count = len(predictions) - high_risk_count
        
        summary = {
            'total_patients': len(df_processed),
            'high_risk_count': high_risk_count,
            'low_risk_count': low_risk_count,
            'high_risk_percentage': float((high_risk_count / len(df_processed)) * 100),
            'average_risk_probability': float(prediction_proba.mean()) * 100  # Convert to percentage
        }
        
        # Feature importance (top 10)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = [
            {
                'feature': EXPECTED_FEATURES[i],
                'importance': float(mean_abs_shap[i])
            }
            for i in np.argsort(mean_abs_shap)[::-1][:10]
        ]
        
        print(f"✅ Predictions complete: {high_risk_count} high-risk, {low_risk_count} low-risk")
        
        return {
            'success': True,
            'summary': summary,
            'feature_importance': feature_importance,
            'patients': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/features")
def get_expected_features():
    """Get list of expected features"""
    return {
        'features': EXPECTED_FEATURES,
        'total_count': len(EXPECTED_FEATURES)
    }

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting Cardiac Readmission Risk API Server v1.1.0")
    print("="*80)
    print("\n📍 Server will be available at: http://localhost:8000")
    print("📍 API docs: http://localhost:8000/docs")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
