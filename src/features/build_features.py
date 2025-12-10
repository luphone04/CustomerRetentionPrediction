import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series: #maps a binary categorical series to 0/1
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals) #get yes/no values

    if valset == {"Yes","No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    
    if valset == {"Male","Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")
    
    #generic bi mapping
    if len(valset) == 2:
        sorted_vals = sorted(vals) #Alphabetical/numerical order
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    

    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy() #avoid modifying original
    print(f"Start feature engineering on {df.shape[1]} columns...")

    # Find categorical columns(dtype) excluding target
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col] # ["Gender", "Partner", "Contract", ...]
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist() # ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    print(f"Found {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns.")

    # Split Categorical by Cardinality
    # Binary features get binary encoding (0/1)
    # Multi-class get one-hot encoding (with drop_first to avoid dummy trap)
    binary_cols =[c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f"Binary features: {len(binary_cols)}, Multi-class features: {len(multi_cols)}")

    if binary_cols:
        print(f"Applying binary encoding to columns: {binary_cols}")
    if multi_cols:
        print(f"Applying one-hot encoding to columns: {multi_cols}")

    #Apply binary encoding

    for c in binary_cols: #
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str)) #convert to str to avoid issues
        print(f"Encoded binary column '{c}' from {original_dtype} to {df[c].dtype}")

    #Convert boolean colyumn as XGBoost requires int/bool types
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    #Apply one-hot encoding
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-class columns: {multi_cols}")
        original_shape = df.shape

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        new_features = df.shape[1] - original_shape[1] + len(multi_cols) #new cols - old cols + dropped firsts
        print(f"Created {new_features} new features from {len(multi_cols)} multi-class columns.")

        for c in binary_cols:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].fillna(0).astype(int) #fill NaNs if any

        print(f"Feature engineering completed. Final dataset has {df.shape[1]} columns.")
        return df

    