# Data validation using pandas (simple, no external dependencies)

import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation...")

    failed_checks = []

    print("Validating schema and required columns...")

    required_columns = [
        "customerID", "Churn", "TotalCharges", "MonthlyCharges", "tenure",
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "Contract"
    ]

    for col in required_columns:
        if col not in df.columns:
            failed_checks.append(f"missing_column:{col}")
            print(f"  ❌ Missing column: {col}")
        else:
            print(f"  ✅ Column exists: {col}")

    if failed_checks:
        return False, failed_checks

    if df["customerID"].isnull().any():
        null_count = df["customerID"].isnull().sum()
        failed_checks.append(f"null_customerID:{null_count}")
        print(f"  ❌ customerID has {null_count} null values")

    print("Validating business logic rules...")

    valid_genders = {"Male", "Female"}
    actual_genders = set(df["gender"].dropna().unique())
    if not actual_genders.issubset(valid_genders):
        invalid = actual_genders - valid_genders
        failed_checks.append(f"invalid_gender:{invalid}")
        print(f"  ❌ Invalid gender values: {invalid}")
    else:
        print(f"  ✅ Gender values valid")

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    valid_yes_no = {"Yes", "No"}

    for col in yes_no_cols:
        if col in df.columns:
            actual_vals = set(df[col].dropna().unique())
            if not actual_vals.issubset(valid_yes_no):
                invalid = actual_vals - valid_yes_no
                failed_checks.append(f"invalid_{col}:{invalid}")
                print(f"  ❌ Invalid {col} values: {invalid}")
            else:
                print(f"  ✅ {col} values valid")

    valid_contracts = {"Month-to-month", "One year", "Two year"}
    actual_contracts = set(df["Contract"].dropna().unique())
    if not actual_contracts.issubset(valid_contracts):
        invalid = actual_contracts - valid_contracts
        failed_checks.append(f"invalid_contract:{invalid}")
        print(f"  ❌ Invalid Contract values: {invalid}")
    else:
        print(f"  ✅ Contract values valid")

    valid_internet = {"DSL", "Fiber optic", "No"}
    actual_internet = set(df["InternetService"].dropna().unique())
    if not actual_internet.issubset(valid_internet):
        invalid = actual_internet - valid_internet
        failed_checks.append(f"invalid_internet:{invalid}")
        print(f"  ❌ Invalid InternetService values: {invalid}")
    else:
        print(f"  ✅ InternetService values valid")

    print("Validating numeric ranges...")

    if (df["tenure"] < 0).any():
        neg_count = (df["tenure"] < 0).sum()
        failed_checks.append(f"negative_tenure:{neg_count}")
        print(f"  ❌ tenure has {neg_count} negative values")
    else:
        print(f"  ✅ tenure values valid (non-negative)")

    if (df["MonthlyCharges"] < 0).any():
        neg_count = (df["MonthlyCharges"] < 0).sum()
        failed_checks.append(f"negative_monthly:{neg_count}")
        print(f"  ❌ MonthlyCharges has {neg_count} negative values")
    else:
        print(f"  ✅ MonthlyCharges values valid (non-negative)")

    print("Validating statistical properties...")

    tenure_mean = df["tenure"].mean()
    if not (0 <= tenure_mean <= 120):
        failed_checks.append(f"unreasonable_tenure_mean:{tenure_mean:.2f}")
        print(f"  ❌ tenure mean ({tenure_mean:.2f}) outside expected range [0, 120]")
    else:
        print(f"  ✅ tenure mean ({tenure_mean:.2f}) is reasonable")

    monthly_mean = df["MonthlyCharges"].mean()
    if not (0 <= monthly_mean <= 200):
        failed_checks.append(f"unreasonable_monthly_mean:{monthly_mean:.2f}")
        print(f"  ❌ MonthlyCharges mean ({monthly_mean:.2f}) outside expected range [0, 200]")
    else:
        print(f"  ✅ MonthlyCharges mean ({monthly_mean:.2f}) is reasonable")

    total_checks = 15  # Approximate number of checks
    passed_checks = total_checks - len(failed_checks)

    if len(failed_checks) == 0:
        print(f"\n✅ Data validation passed: {passed_checks}/{total_checks} checks passed.")
        return True, failed_checks
    else:
        print(f"\n❌ Data validation failed: {len(failed_checks)} issues found.")
        print(f"Failed checks: {failed_checks}")
        return False, failed_checks
