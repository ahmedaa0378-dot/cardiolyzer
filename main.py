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
    title="Cardiolyzer API",
    description="AI-powered cardiac readmission prediction with SHAP explainability — v2 Canonical Schema",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL ARTIFACTS (v2)
# ============================================================================
print("=" * 60)
print("CARDIOLYZER v2 — Loading model artifacts...")
print("=" * 60)

try:
    with open('readmission_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded (v2)")

    with open('scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded (v2)")

    with open('feature_config_v2.pkl', 'rb') as f:
        feature_config = pickle.load(f)
    print("✅ Feature config loaded (v2)")

    with open('shap_explainer_v2.pkl', 'rb') as f:
        shap_explainer = pickle.load(f)
    print("✅ SHAP explainer loaded (v2)")

    EXPECTED_FEATURES = feature_config['feature_names']
    IDENTITY_COLUMNS = feature_config.get('identity_columns', [])
    SHAP_DISPLAY_NAMES = feature_config.get('shap_display_names', {})
    SCHEMA_VERSION = feature_config.get('schema_version', '2.0')

    print(f"✅ Schema version: {SCHEMA_VERSION}")
    print(f"✅ Expected features: {len(EXPECTED_FEATURES)}")
    print(f"✅ SHAP display names: {len(SHAP_DISPLAY_NAMES)} mapped")

except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")
    raise


# ============================================================================
# INTELLIGENT MAPPER — Maps any EMR format to Canonical Schema
# ============================================================================

# Known column aliases for different EMR systems
COLUMN_ALIASES = {
    # Demographics
    'age': ['age', 'age_at_arrival', 'patient_age', 'age_years', 'Age'],
    'sex': ['sex', 'gender', 'Gender', 'patient_sex', 'gender_code'],
    'race_white': ['race_white'],
    'race_black': ['race_black'],
    'race_hispanic': ['race_hispanic'],
    'race_other': ['race_other'],
    'preferred_language_english': ['preferred_language_english'],
    'marital_status_married': ['marital_status_married'],
    'insurance_medicare': ['insurance_medicare'],
    'insurance_medicaid': ['insurance_medicaid'],
    'insurance_commercial': ['insurance_commercial'],
    'has_pcp': ['has_pcp', 'pcp', 'PCP', 'has_primary_care'],

    # Encounter
    'admit_source_ed': ['admit_source_ed', 'ed_admission'],
    'admit_source_transfer': ['admit_source_transfer'],
    'triage_acuity': ['triage_acuity', 'esi_level', 'acuity'],
    'los_ed_hours': ['los_ed_hours', 'ed_los_hours'],
    'los_inpatient_days': ['los_inpatient_days', 'length_of_stay', 'Length_of_stay', 'los_days'],
    'icu_stay': ['icu_stay', 'icu_flag', 'icu_stay_flag'],
    'discharge_to_home': ['discharge_to_home'],
    'discharge_to_snf': ['discharge_to_snf'],
    'discharge_to_rehab': ['discharge_to_rehab'],

    # Diagnosis
    'hf_type_systolic': ['hf_type_systolic', 'hfref', 'systolic_hf'],
    'hf_type_diastolic': ['hf_type_diastolic', 'hfpef', 'diastolic_hf'],
    'hf_type_combined': ['hf_type_combined'],
    'hf_acute_decompensated': ['hf_acute_decompensated', 'acute_decompensation'],
    'secondary_dx_count': ['secondary_dx_count', 'diagnosis_count'],

    # Comorbidities
    'comorbid_hypertension': ['comorbid_hypertension', 'hypertension', 'Hypertension', 'htn_flag'],
    'comorbid_diabetes': ['comorbid_diabetes', 'diabetes', 'Diabetes_mellitus_Type_2', 'diabetes_flag', 'dm_flag'],
    'comorbid_ckd': ['comorbid_ckd', 'ckd', 'Chronic_kidney_disease', 'ckd_flag'],
    'comorbid_copd': ['comorbid_copd', 'copd', 'COPD', 'copd_flag'],
    'comorbid_afib': ['comorbid_afib', 'afib', 'Atrial_fibrillation', 'afib_flag'],
    'comorbid_cad': ['comorbid_cad', 'cad', 'Coronary_artery_disease'],
    'comorbid_depression': ['comorbid_depression', 'depression', 'depression_flag'],
    'comorbid_obesity': ['comorbid_obesity', 'obesity', 'obesity_flag'],
    'comorbid_anemia': ['comorbid_anemia', 'anemia', 'anemia_flag'],
    'comorbid_stroke_tia': ['comorbid_stroke_tia', 'stroke', 'CVA_Stroke_TIA', 'stroke_tia'],
    'comorbid_substance_use': ['comorbid_substance_use', 'substance_use', 'substance_use_flag'],
    'charlson_index': ['charlson_index', 'charlson_score', 'cci'],

    # Vitals
    'vital_heart_rate': ['vital_heart_rate', 'vital_hr_first', 'HR', 'heart_rate', 'hr_first'],
    'vital_sbp': ['vital_sbp', 'vital_sbp_first', 'SBP', 'systolic_bp', 'sbp_first'],
    'vital_dbp': ['vital_dbp', 'vital_dbp_first', 'DBP', 'diastolic_bp', 'dbp_first'],
    'vital_resp_rate': ['vital_resp_rate', 'vital_rr_first', 'resp_rate', 'rr'],
    'vital_spo2': ['vital_spo2', 'vital_spo2_first', 'O2_saturation', 'spo2', 'o2_sat'],
    'vital_temp_c': ['vital_temp_c', 'vital_temp_first_c', 'temperature', 'temp_c'],
    'vital_bmi': ['vital_bmi', 'BMI', 'bmi'],
    'vital_weight_kg': ['vital_weight_kg', 'weight_kg', 'weight'],

    # Labs
    'lab_bnp': ['lab_bnp', 'bnp_value', 'BNP', 'bnp'],
    'lab_nt_probnp': ['lab_nt_probnp', 'NT_Pro_BNP', 'nt_probnp', 'ntbnp'],
    'lab_troponin': ['lab_troponin', 'troponin_first', 'Troponin', 'troponin', 'trop'],
    'lab_creatinine': ['lab_creatinine', 'creatinine_first', 'creatinine', 'creat'],
    'lab_bun': ['lab_bun', 'bun_first', 'bun', 'BUN'],
    'lab_egfr': ['lab_egfr', 'GFR', 'gfr', 'egfr', 'estimated_gfr'],
    'lab_sodium': ['lab_sodium', 'sodium_first', 'Sodium', 'sodium', 'na'],
    'lab_potassium': ['lab_potassium', 'potassium_first', 'Potassium', 'potassium', 'k'],
    'lab_hemoglobin': ['lab_hemoglobin', 'hemoglobin_first', 'hemoglobin', 'hgb'],
    'lab_wbc': ['lab_wbc', 'wbc_first', 'wbc', 'WBC'],
    'lab_albumin': ['lab_albumin', 'albumin', 'alb'],

    # Cardiac
    'ejection_fraction': ['ejection_fraction', 'Ejectin_Fraction', 'ef', 'lvef', 'ef_percent'],
    'nyha_class': ['nyha_class', 'NYHA_class_1-4', 'nyha', 'nyha_functional_class'],
    'ecg_abnormal': ['ecg_abnormal', 'ecg_result_flag', 'ekg_abnormal'],
    'cxr_pulmonary_edema': ['cxr_pulmonary_edema', 'cxr_result_flag', 'pulmonary_edema'],
    'has_icd_device': ['has_icd_device', 'ICD', 'icd_device', 'icd_crt'],

    # Medications
    'med_beta_blocker': ['med_beta_blocker', 'beta_blocker_on_discharge', 'Medications-Betablocker', 'betablocker'],
    'med_acei_arb_arni': ['med_acei_arb_arni', 'ace_arb_arni_on_discharge', 'ACE_inhibitor', 'acei_arb'],
    'med_mra': ['med_mra', 'mra_on_discharge', 'MRA', 'mra', 'spironolactone'],
    'med_sglt2i': ['med_sglt2i', 'sglt2_on_discharge', 'SGLT2i', 'sglt2i', 'sglt2'],
    'med_loop_diuretic': ['med_loop_diuretic', 'iv_loop_diuretic_given', 'loop_diuretic', 'diuretic'],
    'med_statin': ['med_statin', 'Statin', 'statin'],
    'med_anticoagulant': ['med_anticoagulant', 'anticoagulant', 'anticoag'],
    'gdmt_adherence_score': ['gdmt_adherence_score', 'gdmt_score'],

    # Discharge Planning
    'med_reconciliation_done': ['med_reconciliation_done', 'med_reconciliation_completed'],
    'hf_education_done': ['hf_education_done', 'hf_education_completed'],
    'followup_scheduled': ['followup_scheduled', 'followup_appt_scheduled'],
    'followup_days': ['followup_days', 'days_to_followup'],
    'cardiology_consult': ['cardiology_consult', 'consult_cardiology'],
    'case_management_consult': ['case_management_consult'],
    'social_work_consult': ['social_work_consult'],
    'discharge_note_signed': ['discharge_note_signed', 'discharge_note_signed_flag'],

    # Prior Utilization
    'prior_ed_visits_6m': ['prior_ed_visits_6m', 'Number_of_ER_visits', 'ed_visits_6m'],
    'prior_admissions_6m': ['prior_admissions_6m', 'admissions_6m'],
    'prior_admissions_12m': ['prior_admissions_12m', 'Number_of_Hospitla_visits', 'admissions_12m', 'Admissions_Last_12_Months'],
    'days_since_last_admission': ['days_since_last_admission'],
    'prior_hf_admissions_12m': ['prior_hf_admissions_12m'],

    # Social Determinants
    'lives_alone': ['lives_alone', 'Lives_Alone'],
    'housing_instability': ['housing_instability', 'housing_instability_flag'],
    'transportation_barrier': ['transportation_barrier', 'Transportation_Access'],
    'smoking_current': ['smoking_current', 'current_smoker'],
    'smoking_former': ['smoking_former', 'former_smoker'],
    'alcohol_heavy_use': ['alcohol_heavy_use', 'substance_use_flag'],
}

# Columns that identify patients (extracted before preprocessing, not sent to model)
IDENTITY_PATTERNS = [
    'patient_mrn', 'mrn', 'encounter_id', 'account_id', 'patient_name', 'name',
    'dob', 'date_of_birth', 'attending_provider', 'attending_physician',
    'admission_datetime', 'date_of_admission', 'discharge_datetime',
    'date_of_discharge', 'department', 'department_name', 'room_bed',
    'pcp_id', 'attending_provider_id', 'zip_code', 'address',
    # v1 columns to drop
    'paitient_id', 'high_risk_flag', 'risk_score_(0-10)',
    'readmiited_within_30_days', 'readmitted_within_30_days',
    'readmission_30d', 'days_to_readmission',
    # Enhanced columns (UI display only, not model features)
    'admission_type', 'chief_complaint', 'smoking_status', 'alcohol_use',
    'social_support', 'transportation_access', 'medication_adherence',
    'admissions_last_12_months', 'previous_admission_dates',
    'discharge_disposition', 'scheduled_followup_date', 'primary_language',
    'care_team_notes', 'gender_identity', 'ethnicity', 'arrival_mode',
    'o2_device_initial', 'cxr_result_flag', 'ecg_result_flag',
    'arrival_datetime', 'triage_datetime', 'ed_departure_datetime',
    'death_during_encounter_flag', 'los_ed_hours_raw',
]


def intelligent_map(df_raw):
    """
    Intelligent Mapper: Translates any EMR format into Canonical Schema.
    
    1. Extract identity/metadata columns
    2. Match incoming columns to canonical feature names
    3. Transform values (Y/n → 0/1, text → binary flags, etc.)
    4. Calculate derived features (Charlson, eGFR, GDMT score)
    5. Fill missing features with 0
    6. Return canonical DataFrame + metadata
    """
    
    df = df_raw.copy()
    
    # Clean column names for matching
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
    
    # ----------------------------------------------------------------
    # STEP 1: Extract patient metadata BEFORE any processing
    # ----------------------------------------------------------------
    metadata = extract_patient_metadata(df_raw)
    
    # ----------------------------------------------------------------
    # STEP 2: Identify and drop non-feature columns
    # ----------------------------------------------------------------
    clean_cols_lower = {col: col.lower() for col in df.columns}
    cols_to_drop = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in IDENTITY_PATTERNS):
            cols_to_drop.append(col)
    
    # Also drop any remaining object columns that aren't mappable
    df_work = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # ----------------------------------------------------------------
    # STEP 3: Map columns to canonical names
    # ----------------------------------------------------------------
    canonical_df = pd.DataFrame(index=range(len(df)))
    mapped_count = 0
    
    for canonical_name, aliases in COLUMN_ALIASES.items():
        matched = False
        for alias in aliases:
            # Try exact match first
            if alias in df_work.columns:
                canonical_df[canonical_name] = df_work[alias].values
                mapped_count += 1
                matched = True
                break
            # Try cleaned name match
            alias_clean = alias.strip().replace(' ', '_').replace('/', '_')
            if alias_clean in df_work.columns:
                canonical_df[canonical_name] = df_work[alias_clean].values
                mapped_count += 1
                matched = True
                break
        
        if not matched:
            canonical_df[canonical_name] = 0  # Default for missing features
    
    print(f"   📊 Mapped {mapped_count}/{len(COLUMN_ALIASES)} features from input data")
    
    # ----------------------------------------------------------------
    # STEP 4: Transform values to canonical format
    # ----------------------------------------------------------------
    
    # Convert any remaining Y/n/Yes/No strings to 0/1
    binary_map = {'Y': 1, 'y': 1, 'N': 0, 'n': 0, 'Yes': 1, 'yes': 1, 'No': 0, 'no': 0,
                  'True': 1, 'true': 1, 'False': 0, 'false': 0}
    
    for col in canonical_df.columns:
        if canonical_df[col].dtype == 'object':
            # Try binary mapping first
            mapped = canonical_df[col].map(binary_map)
            if mapped.notna().sum() > len(canonical_df) * 0.5:  # If >50% mapped successfully
                canonical_df[col] = mapped.fillna(0).astype(int)
            else:
                # Try numeric conversion
                canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0)
    
    # Ensure all columns are numeric
    for col in canonical_df.columns:
        canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0)
    
    # ----------------------------------------------------------------
    # STEP 5: Handle special transformations
    # ----------------------------------------------------------------
    
    # Handle sex/gender: convert M/F to 0/1 if needed
    if 'sex' in canonical_df.columns:
        sex_col = canonical_df['sex']
        # If values are strings like 'M'/'F', they would have been converted to 0 above
        # Check the original data for proper mapping
        for alias in COLUMN_ALIASES.get('sex', []):
            if alias in df_work.columns:
                orig_vals = df_work[alias].astype(str).str.strip().str.upper()
                canonical_df['sex'] = orig_vals.map({'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0, '1': 1, '0': 0}).fillna(0).astype(int)
                break
    
    # Handle race: if we have a single race column, one-hot encode it
    race_cols_present = sum(1 for c in ['race_white', 'race_black', 'race_hispanic', 'race_other'] 
                           if canonical_df[c].sum() > 0)
    if race_cols_present == 0:
        # Check if there's a raw race column
        for col in df_work.columns:
            if col.lower() in ['race', 'race_ethnicity']:
                race_vals = df_work[col].astype(str).str.strip().str.upper()
                canonical_df['race_white'] = race_vals.isin(['WHITE', 'W', 'CAUCASIAN']).astype(int)
                canonical_df['race_black'] = race_vals.isin(['BLACK', 'B', 'AFRICAN AMERICAN', 'AA']).astype(int)
                canonical_df['race_hispanic'] = race_vals.isin(['HISPANIC', 'H', 'LATINO']).astype(int)
                canonical_df['race_other'] = (~race_vals.isin(['WHITE', 'W', 'CAUCASIAN', 'BLACK', 'B', 
                    'AFRICAN AMERICAN', 'AA', 'HISPANIC', 'H', 'LATINO'])).astype(int)
                break
    
    # Handle insurance: if we have a single insurance column, one-hot encode it
    ins_cols_present = sum(1 for c in ['insurance_medicare', 'insurance_medicaid', 'insurance_commercial'] 
                          if canonical_df[c].sum() > 0)
    if ins_cols_present == 0:
        for col in df_work.columns:
            if col.lower() in ['insurance', 'insurance_payer', 'payer']:
                ins_vals = df_work[col].astype(str).str.strip().str.upper()
                canonical_df['insurance_medicare'] = ins_vals.str.contains('MEDICARE', na=False).astype(int)
                canonical_df['insurance_medicaid'] = ins_vals.str.contains('MEDICAID', na=False).astype(int)
                canonical_df['insurance_commercial'] = (~ins_vals.str.contains('MEDICARE|MEDICAID', na=False)).astype(int)
                break
    
    # Handle discharge disposition: if we have a single column, split into flags
    disp_cols_present = sum(1 for c in ['discharge_to_home', 'discharge_to_snf', 'discharge_to_rehab'] 
                           if canonical_df[c].sum() > 0)
    if disp_cols_present == 0:
        for col in df_work.columns:
            if col.lower() in ['discharge_disposition', 'discharge_status']:
                disp_vals = df_work[col].astype(str).str.strip().str.upper()
                canonical_df['discharge_to_home'] = disp_vals.str.contains('HOME', na=False).astype(int)
                canonical_df['discharge_to_snf'] = disp_vals.str.contains('SNF|SKILLED|NURSING', na=False).astype(int)
                canonical_df['discharge_to_rehab'] = disp_vals.str.contains('REHAB', na=False).astype(int)
                break
    
    # Handle HF type: if we have a diagnosis text, derive HF flags
    hf_cols_present = sum(1 for c in ['hf_type_systolic', 'hf_type_diastolic', 'hf_type_combined'] 
                         if canonical_df[c].sum() > 0)
    if hf_cols_present == 0:
        for col in df_work.columns:
            if col.lower() in ['primary_diagnosis', 'ed_primary_diagnosis_text', 'diagnosis']:
                dx_vals = df_work[col].astype(str).str.strip().str.lower()
                canonical_df['hf_type_systolic'] = dx_vals.str.contains('reduced|systolic|hfref', na=False).astype(int)
                canonical_df['hf_type_diastolic'] = dx_vals.str.contains('preserved|diastolic|hfpef', na=False).astype(int)
                canonical_df['hf_type_combined'] = ((canonical_df['hf_type_systolic'] == 0) & 
                    (canonical_df['hf_type_diastolic'] == 0)).astype(int)
                canonical_df['hf_acute_decompensated'] = dx_vals.str.contains('acute|decompensated', na=False).astype(int)
                break
    
    # Handle smoking: if we have a single smoking status column
    if canonical_df['smoking_current'].sum() == 0 and canonical_df['smoking_former'].sum() == 0:
        for col in df_work.columns:
            if col.lower() in ['smoking_status']:
                smoke_vals = df_work[col].astype(str).str.strip().str.lower()
                canonical_df['smoking_current'] = smoke_vals.str.contains('current', na=False).astype(int)
                canonical_df['smoking_former'] = smoke_vals.str.contains('former', na=False).astype(int)
                break
    
    # ----------------------------------------------------------------
    # STEP 6: Calculate derived features
    # ----------------------------------------------------------------
    
    # GDMT adherence score (if not already set)
    if canonical_df['gdmt_adherence_score'].sum() == 0:
        canonical_df['gdmt_adherence_score'] = (
            canonical_df['med_beta_blocker'] + 
            canonical_df['med_acei_arb_arni'] + 
            canonical_df['med_mra'] + 
            canonical_df['med_sglt2i']
        )
    
    # eGFR from creatinine if eGFR is missing but creatinine is present
    if canonical_df['lab_egfr'].sum() == 0 and canonical_df['lab_creatinine'].sum() > 0:
        creat = canonical_df['lab_creatinine'].clip(lower=0.3)
        age_vals = canonical_df['age'].clip(lower=18)
        sex_vals = canonical_df['sex']
        canonical_df['lab_egfr'] = np.clip(
            175 * (creat ** -1.154) * (age_vals ** -0.203) * np.where(sex_vals == 0, 0.742, 1.0),
            3, 120
        ).astype(int)
    
    # Charlson index approximation if not present
    if canonical_df['charlson_index'].sum() == 0:
        comorbidity_sum = sum(canonical_df[c] for c in [
            'comorbid_hypertension', 'comorbid_diabetes', 'comorbid_ckd',
            'comorbid_copd', 'comorbid_afib', 'comorbid_cad',
            'comorbid_depression', 'comorbid_obesity', 'comorbid_anemia',
            'comorbid_stroke_tia', 'comorbid_substance_use'
        ])
        age_factor = np.where(canonical_df['age'] >= 80, 4,
            np.where(canonical_df['age'] >= 70, 3,
            np.where(canonical_df['age'] >= 60, 2, 1)))
        canonical_df['charlson_index'] = np.clip(comorbidity_sum + age_factor, 0, 15)
    
    # ----------------------------------------------------------------
    # STEP 7: Fill missing values and ensure correct feature order
    # ----------------------------------------------------------------
    for col in canonical_df.columns:
        if canonical_df[col].isnull().sum() > 0:
            canonical_df[col].fillna(canonical_df[col].median() if canonical_df[col].sum() != 0 else 0, inplace=True)
    
    # Ensure ALL expected features are present in correct order
    for feature in EXPECTED_FEATURES:
        if feature not in canonical_df.columns:
            canonical_df[feature] = 0
    
    canonical_df = canonical_df[EXPECTED_FEATURES]
    
    return canonical_df, metadata


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_patient_metadata(df):
    """Extract patient metadata (ID, name, age, gender, diagnosis) before preprocessing"""
    
    metadata = []
    df_cols = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
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
        for pattern in ['patient_mrn', 'patient_id', 'paitient_id', 'encounter_id', 'mrn']:
            matches = [orig for clean, orig in col_mapping.items() if pattern in clean]
            if matches:
                val = df.iloc[i][matches[0]]
                if pd.notna(val):
                    patient_meta['patient_id'] = str(val)
                break
        
        # Extract Patient Name
        for pattern in ['patient_name', 'name']:
            matches = [orig for clean, orig in col_mapping.items() if pattern in clean and 'user' not in clean]
            if matches:
                val = df.iloc[i][matches[0]]
                if pd.notna(val):
                    patient_meta['patient_name'] = str(val)
                break
        
        # Extract Age
        for pattern in ['age', 'age_at_arrival']:
            matches = [orig for clean, orig in col_mapping.items() if clean == pattern or clean.endswith('_age')]
            if matches:
                val = df.iloc[i][matches[0]]
                if pd.notna(val):
                    try:
                        patient_meta['age'] = int(float(val))
                    except:
                        pass
                break
        
        # Extract Gender
        for pattern in ['gender', 'sex', 'gender_identity']:
            matches = [orig for clean, orig in col_mapping.items() if pattern in clean]
            if matches:
                val = df.iloc[i][matches[0]]
                if pd.notna(val):
                    patient_meta['gender'] = str(val)
                break
        
        # Extract Primary Diagnosis
        for pattern in ['primary_diagnosis', 'ed_primary_diagnosis_text', 'diagnosis']:
            matches = [orig for clean, orig in col_mapping.items() if pattern in clean.replace('_', ' ') or pattern in clean]
            if matches:
                val = df.iloc[i][matches[0]]
                if pd.notna(val):
                    patient_meta['primary_diagnosis'] = str(val)
                break
        
        # Fallbacks
        if patient_meta['patient_id'] is None:
            patient_meta['patient_id'] = f"P{str(i + 1).zfill(6)}"
        if patient_meta['patient_name'] is None:
            patient_meta['patient_name'] = f"Patient {patient_meta['patient_id']}"
        
        metadata.append(patient_meta)
    
    return metadata


def get_display_name(feature_name):
    """Get clinician-friendly display name for a feature"""
    return SHAP_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' ').title())


def generate_patient_explanation(patient_idx, shap_values, features):
    """Generate human-readable explanation with display names"""
    
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
                'feature': get_display_name(row['feature']),
                'feature_raw': row['feature'],
                'value': float(row['feature_value']),
                'impact': float(row['shap_value'])
            }
            for _, row in top_positive.iterrows()
        ],
        'protective_factors': [
            {
                'feature': get_display_name(row['feature']),
                'feature_raw': row['feature'],
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
    return {
        "status": "healthy",
        "message": "Cardiolyzer API v2 — Canonical Schema",
        "version": "2.0.0",
        "schema_version": SCHEMA_VERSION,
        "model_loaded": True,
        "shap_enabled": True,
        "features": len(EXPECTED_FEATURES)
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded",
        "scaler": "loaded",
        "shap_explainer": "loaded",
        "expected_features": len(EXPECTED_FEATURES),
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict")
async def predict_readmission(file: UploadFile = File(...)):
    """
    Predict readmission risk for uploaded patient data.
    Accepts CSV or Excel files in ANY EMR format — the Intelligent Mapper
    translates to canonical schema automatically.
    """
    
    try:
        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or Excel file.")
        
        # Read file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        else:
            from io import BytesIO
            df = pd.read_excel(BytesIO(contents))
        
        print(f"\n{'='*60}")
        print(f"📊 Received: {file.filename}")
        print(f"   Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        print(f"   Columns: {df.columns.tolist()[:10]}...")
        print(f"{'='*60}")
        
        # Validate
        if df.shape[0] > 10000:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10,000 patients.")
        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="File is empty.")
        
        # ---- INTELLIGENT MAPPER ----
        print("🔄 Running Intelligent Mapper...")
        df_canonical, patient_metadata = intelligent_map(df)
        print(f"   ✅ Canonical schema: {df_canonical.shape[1]} features")
        
        # Scale features
        X_scaled = scaler.transform(df_canonical)
        
        # Predict
        print("🤖 Running predictions...")
        predictions = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)[:, 1]
        
        # SHAP
        print("🔬 Computing SHAP values...")
        shap_values = shap_explainer.shap_values(X_scaled)
        
        # Build results
        results = []
        for i in range(len(df_canonical)):
            meta = patient_metadata[i]
            
            patient_result = {
                'patient_id': meta['patient_id'],
                'patient_name': meta['patient_name'],
                'age': meta['age'],
                'gender': meta['gender'],
                'primary_diagnosis': meta['primary_diagnosis'],
                'risk_prediction': int(predictions[i]),
                'risk_probability': float(prediction_proba[i]) * 100,
                'risk_level': 'High' if predictions[i] == 1 else 'Low',
                'explanation': generate_patient_explanation(i, shap_values, df_canonical)
            }
            results.append(patient_result)
        
        # Summary
        high_risk_count = int(predictions.sum())
        low_risk_count = len(predictions) - high_risk_count
        
        summary = {
            'total_patients': len(df_canonical),
            'high_risk_count': high_risk_count,
            'low_risk_count': low_risk_count,
            'high_risk_percentage': float((high_risk_count / len(df_canonical)) * 100),
            'average_risk_probability': float(prediction_proba.mean()) * 100
        }
        
        # Feature importance with display names
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = [
            {
                'feature': get_display_name(EXPECTED_FEATURES[i]),
                'feature_raw': EXPECTED_FEATURES[i],
                'importance': float(mean_abs_shap[i])
            }
            for i in np.argsort(mean_abs_shap)[::-1][:15]
        ]
        
        print(f"\n✅ Complete: {high_risk_count} high-risk, {low_risk_count} low-risk")
        print(f"   AUC expected: ~0.845 | Avg risk: {prediction_proba.mean()*100:.1f}%")
        
        return {
            'success': True,
            'summary': summary,
            'feature_importance': feature_importance,
            'patients': results,
            'schema_version': SCHEMA_VERSION,
            'features_mapped': df_canonical.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/features")
def get_expected_features():
    """Get list of expected features with display names"""
    return {
        'features': [
            {
                'name': f,
                'display_name': get_display_name(f)
            }
            for f in EXPECTED_FEATURES
        ],
        'total_count': len(EXPECTED_FEATURES),
        'schema_version': SCHEMA_VERSION
    }


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 Cardiolyzer API v2.0.0 — Canonical Schema")
    print("=" * 60)
    print(f"📍 http://localhost:8000")
    print(f"📍 Docs: http://localhost:8000/docs")
    print(f"📊 Features: {len(EXPECTED_FEATURES)}")
    print(f"🗺️  Schema: {SCHEMA_VERSION}")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
