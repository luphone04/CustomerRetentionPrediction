# After loading the data

import great_expectations as ge
from typing import Tuple, List

def validate_data(df) -> Tuple[ bool, List[str]]: #list str for error messages
    print("Starting data validation with Grweat Expectations...")

    ge_df = ge.dataset.PandasDataset(df)

    print("Validating schema and required columns...")

    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_to_not_be_null("customerID")

    ge_df.expect_column_to_exist("Churn")

    ge_df.expect_column_to_exist("TotalCharges")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("tenure")

    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("MultipleLines")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    print("Validating business logic rules")

    # Gender
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", " Female"])

    # Yes/No columns
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        ge_df.expect_column_values_to_be_in_set(col, ["Yes", "No"])

    #Contract types
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])

    # InternetService types
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    #Numeric Range Validations
    print("Validating numeric ranges...")

    #Tenure should be non-negative
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)

    #MonthlyCharges should be non-negative
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    #TotalCharges should be non-negative
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # Statistical Validations
    print("Validating statistical properties...")

    #tenure mean should be reasonable
    ge_df.expect_column_mean_to_be_between("tenure", min_value=0, max_value=120)

    #MonthlyCharges mean should be reasonable
    ge_df.expect_column_mean_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    #no missing values in key numeric features
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    ge_df.expect_column_values_to_not_be_null("TotalCharges")

    # Check Data Consistency
    print("Validating data consistency checks...")

    #Total Charges should be >= MonthlyCharges (except for new customers)
    ge_df.expect_column_values_to_be_greater_than_or_equal(
        columnA = "TotalCharges", 
        columnB = "MonthlyCharges",
        or_equal=True,
        mostly=True #allow some exceptions for new customers
        )

    #RUN validation suite
    print("Running validation suite...")
    results = ge_df.validate()

    #processs reuslts
    failed_expectations = []
    for res in results["results"]:
        if not res["success"]:
            expectation_type = res["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    #print validfation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"Data validation passed: {passed_checks}/{total_checks} checks passed.")
    else:
        print(f"Data validation failed: {failed_checks}/{total_checks} checks failed.")
        print("Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations