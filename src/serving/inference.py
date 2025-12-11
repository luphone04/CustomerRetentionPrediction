import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(SERVING_DIR, "model", "2aa2e525c0924aaf83270b1ee3534f53", "artifacts")
        local_model_paths = [local_model_path] if os.path.exists(local_model_path) else []
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")


try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")


BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           
    "Partner": {"No": 0, "Yes": 1},               
    "Dependents": {"No": 0, "Yes": 1},            
    "PhoneService": {"No": 0, "Yes": 1},          
    "PaperlessBilling": {"No": 0, "Yes": 1},      
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    
                .str.strip()                  
                .map(mapping)                   
                .astype("Int64")             
                .fillna(0)                     
                .astype(int)                
            )

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:

        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:

    df = pd.DataFrame([input_dict])
    

    df_enc = _serve_transform(df)
    

    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    

    if result == 1:
        return "Likely to churn"      
        return "Not likely to churn"  


if __name__ == "__main__":
    # Example customers
    customers = [
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.50,
            "TotalCharges": 179.00
        },
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 95.00,
            "TotalCharges": 6175.00
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.00,
            "TotalCharges": 70.00
        }
    ]

    print("\n" + "=" * 50)
    print(f"{'Customer':<12} {'Prediction':<25}")
    print("=" * 50)

    for i, cust in enumerate(customers, 1):
        result = predict(cust)
        print(f"Customer {i:<3} {result:<25}")

    print("=" * 50)