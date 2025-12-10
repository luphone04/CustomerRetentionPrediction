import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df.columns = df.columns.str.strip() # Remove leading/trailing spaces from column names

    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    # Convert TotalCharges to numeric, coercing errors to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int) # Fill Nan and convert to int

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

