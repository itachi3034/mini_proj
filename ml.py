import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# Make sure script runs from its own folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 1. Load the Data ---
try:
    df = pd.read_csv('bulk_investment_data.csv')
except FileNotFoundError:
    print("Error: 'bulk_investment_data.csv' not found. Please ensure it's in the correct folder.")
    raise SystemExit

# --- 2. Data Preprocessing: Convert Machine_Model (Text) to Numbers ---
df = pd.get_dummies(df, columns=['Machine_Model'], drop_first=True)

# --- 3. Define Features (X) and Multiple Targets (y) ---
target_cols = [
    'Actual_Lifespan_Yrs',
    'Historical_Total_Maint_Cost',
    'Task_Completion_Time_Hrs'
]
X = df.drop(columns=target_cols)
y = df[target_cols]

# --- 4. Split the Data ---
# (You might want test_size=0.2; keeping your 0.8 for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 5. Initialize and Train the Multi-Output Model ---
base_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
multi_output_model = MultiOutputRegressor(base_estimator)

print("\nStarting model training...")
multi_output_model.fit(X_train, y_train)
print("Model training complete!")

# --- 6. Evaluate the Model (Per Target) ---
y_pred_test = multi_output_model.predict(X_test)

print("\n--- Model Performance on Test Data ---")
for i, target in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
    print(f"Target: {target}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")

# ============================================================
#  USER-DRIVEN PREDICTION + FINANCIAL COMPARISON SECTION
# ============================================================

# Helper: safely get numeric input
def get_float(prompt_text):
    while True:
        try:
            return float(input(prompt_text))
        except ValueError:
            print("Please enter a valid number.")

def get_int(prompt_text):
    while True:
        try:
            return int(input(prompt_text))
        except ValueError:
            print("Please enter a valid integer.")

# --- 7. Collect User Inputs for Machine Options ---
print("\n=======================================================")
print("        USER INPUT: MACHINE OPTIONS TO COMPARE        ")
print("=======================================================")

n_machines = get_int("How many machine options do you want to compare? ")

user_rows = []
for i in range(n_machines):
    print(f"\n--- Machine {i+1} ---")
    machine_name = input("Enter machine model/name (e.g., 'Lathe X-A1'): ").strip()

    initial_cost = get_float("Initial cost (USD): ")
    scrap_value = get_float("Expected scrap/salvage value at end of life (USD): ")
    gst_rate_pct = get_float("GST tax rate (%) on purchase (e.g., 18 for 18%): ")

    power = get_float("Power consumption (kWh per hour): ")
    capacity = get_float("Max capacity (units per hour): ")
    maint_interval = get_float("Scheduled maintenance interval (days): ")
    failure_rate = get_float("Historical failure rate (0–1): ")

    user_rows.append({
        "Machine_Model": machine_name,
        "Initial_Cost_USD": initial_cost,
        "Scrap_Value_USD": scrap_value,
        "GST_Rate_Pct": gst_rate_pct,
        "Power_Consumption_kWh_hr": power,
        "Max_Capacity_Units_hr": capacity,
        "Scheduled_Maint_Interval_Days": maint_interval,
        "Historical_Failure_Rate": failure_rate
    })

new_machine_data = pd.DataFrame(user_rows)

# --- 8. Use the ML model to predict outputs for user machines ---

# Columns used by the model (must match training features)
model_input_cols = [
    "Initial_Cost_USD",
    "Power_Consumption_kWh_hr",
    "Max_Capacity_Units_hr",
    "Scheduled_Maint_Interval_Days",
    "Historical_Failure_Rate",
    "Machine_Model"
]

# Make a copy of just the columns the model expects
model_input_df = new_machine_data[model_input_cols].copy()

# Apply same encoding as training
model_input_df = pd.get_dummies(model_input_df, columns=['Machine_Model'], drop_first=True)

# Align columns with training data (X)
model_input_df = model_input_df.reindex(columns=X.columns, fill_value=0)

# Predict
predicted_outputs = multi_output_model.predict(model_input_df)
predictions_df = pd.DataFrame(predicted_outputs, columns=target_cols)

# Merge predictions back with user data
comparison_df = new_machine_data.copy()
comparison_df['Predicted_Lifespan'] = predictions_df['Actual_Lifespan_Yrs']
comparison_df['Predicted_Maint_Cost'] = predictions_df['Historical_Total_Maint_Cost']
comparison_df['Predicted_Task_Time'] = predictions_df['Task_Completion_Time_Hrs']

print("\n--- ML Predicted Outputs for Your Inputs ---")
print(predictions_df.assign(Machine_Model=new_machine_data["Machine_Model"]).to_string(index=False))

# --- 9. Get Financial Parameters from User ---
print("\n=======================================================")
print("      USER INPUT: FINANCIAL PARAMETERS (GLOBAL)       ")
print("=======================================================")

BULK_QUANTITY = get_int("Bulk quantity of units to purchase: ")
TIME_HORIZON_YRS = get_int("Time horizon for comparison (years): ")
TARGET_REVENUE_PER_UNIT_YR = get_float("Expected annual revenue per unit (USD): ")
ANNUAL_OPERATING_COST_USD = get_float("Annual operating cost per unit (USD): ")

# --- 10. Financial Metrics Calculation ---

# Effective initial cost including GST and scrap:
# Effective Initial = Initial_Cost * (1 + GST%) - Scrap_Value
comparison_df['GST_Factor'] = 1 + (comparison_df['GST_Rate_Pct'] / 100.0)
comparison_df['Effective_Initial_Cost'] = (
    comparison_df['Initial_Cost_USD'] * comparison_df['GST_Factor']
    - comparison_df['Scrap_Value_USD']
)

# Annualized maintenance cost (based on predicted lifespan)
comparison_df['Annual_Maint_Cost'] = (
    comparison_df['Predicted_Maint_Cost'] / comparison_df['Predicted_Lifespan']
)

# Total Cost of Ownership (TCO) over time horizon
comparison_df['TCO_Per_Unit'] = (
    comparison_df['Effective_Initial_Cost']
    + (ANNUAL_OPERATING_COST_USD * TIME_HORIZON_YRS)
    + (comparison_df['Annual_Maint_Cost'] * TIME_HORIZON_YRS)
)

# Total revenue & net profit
comparison_df['Total_Revenue_Per_Unit'] = TARGET_REVENUE_PER_UNIT_YR * TIME_HORIZON_YRS
comparison_df['Net_Profit_Per_Unit'] = (
    comparison_df['Total_Revenue_Per_Unit'] - comparison_df['TCO_Per_Unit']
)

# ROI
comparison_df['ROI_5Yrs'] = (
    comparison_df['Net_Profit_Per_Unit'] /
    comparison_df['Effective_Initial_Cost']
) * 100.0

# Scale to bulk quantity
comparison_df['Bulk_Total_TCO'] = comparison_df['TCO_Per_Unit'] * BULK_QUANTITY
comparison_df['Bulk_Total_Profit'] = comparison_df['Net_Profit_Per_Unit'] * BULK_QUANTITY

# --- 11. Final Recommendation ---

best_model_row = comparison_df.loc[comparison_df['Net_Profit_Per_Unit'].idxmax()]

print("\n=======================================================")
print("  💰 FINANCIAL COMPARISON (USER-DEFINED INPUTS) 💰")
print("=======================================================")
print(
    f"ASSUMPTIONS: Bulk Quantity={BULK_QUANTITY}, "
    f"Time Horizon={TIME_HORIZON_YRS} years, "
    f"Annual Revenue/Unit=${TARGET_REVENUE_PER_UNIT_YR:,.2f}, "
    f"Annual Operating Cost/Unit=${ANNUAL_OPERATING_COST_USD:,.2f}"
)

# Build a nice summary table
summary_cols = [
    'Machine_Model',
    'Initial_Cost_USD',
    'Scrap_Value_USD',
    'GST_Rate_Pct',
    'Predicted_Lifespan',
    'Annual_Maint_Cost',
    'Net_Profit_Per_Unit',
    'ROI_5Yrs'
]

summary = comparison_df[summary_cols].copy()
summary.rename(columns={
    'Machine_Model': 'Machine',
    'Initial_Cost_USD': 'Initial Cost',
    'Scrap_Value_USD': 'Scrap Value',
    'GST_Rate_Pct': 'GST (%)',
    'Predicted_Lifespan': 'Pred. Life (Yrs)',
    'Annual_Maint_Cost': 'Annual Maint.',
    'Net_Profit_Per_Unit': 'Net Profit/Unit',
    'ROI_5Yrs': 'ROI (%)'
}, inplace=True)

summary['Initial Cost'] = summary['Initial Cost'].map('${:,.0f}'.format)
summary['Scrap Value'] = summary['Scrap Value'].map('${:,.0f}'.format)
summary['Annual Maint.'] = summary['Annual Maint.'].map('${:,.0f}'.format)
summary['Net Profit/Unit'] = summary['Net Profit/Unit'].map('${:,.0f}'.format)
summary['ROI (%)'] = summary['ROI (%)'].map('{:.2f}%'.format)

print("\n--- Comparative Financial Metrics (Per Unit) ---")
print(summary.to_string(index=False))

print("\n--- FINAL INVESTMENT RECOMMENDATION ---")
print(f"Recommended machine for a bulk investment of {BULK_QUANTITY} units: {best_model_row['Machine_Model']}")
print(
    f"Projected Net Profit over {TIME_HORIZON_YRS} years "
    f"(bulk): ${best_model_row['Bulk_Total_Profit']:,.0f}"
)
print(f"Estimated ROI over {TIME_HORIZON_YRS} years: {best_model_row['ROI_5Yrs']:.2f}%")
print("=======================================================")
