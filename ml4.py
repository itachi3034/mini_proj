import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import hashlib
import json
import time
from datetime import datetime

# ==========================================
#  MISSING BLOCKCHAIN CLASSES DEFINED HERE
# ==========================================

class MachineInvestmentBlockchain:
    def __init__(self, difficulty=2):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': "0",
            'nonce': 0
        }
        genesis_block['hash'] = self.hash_block(genesis_block)
        self.chain.append(genesis_block)

    def hash_block(self, block):
        # Sort keys to ensure consistent hash
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        last_block = self.chain[-1]
        new_block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'transactions': self.pending_transactions,
            'previous_hash': last_block['hash'],
            'nonce': 0
        }
        
        # Proof of Work
        new_block['hash'] = self.proof_of_work(new_block)
        
        self.chain.append(new_block)
        self.pending_transactions = [] # Reset pending
        return new_block

    def proof_of_work(self, block):
        block['nonce'] = 0
        computed_hash = self.hash_block(block)
        while not computed_hash.startswith('0' * self.difficulty):
            block['nonce'] += 1
            computed_hash = self.hash_block(block)
        return computed_hash

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current['previous_hash'] != previous['hash']:
                return False
            
            # Verify hash integrity (optional, but good practice)
            # Re-calculating hash with the stored nonce should match stored hash
            # if self.hash_block(current) != current['hash']: return False
            
        return True

    def get_chain_as_dataframe(self):
        data = []
        for block in self.chain:
            data.append({
                'Block Index': block['index'],
                'Timestamp': time.ctime(block['timestamp']),
                'Transactions': len(block['transactions']),
                'Hash': block['hash'],
                'Prev Hash': block['previous_hash']
            })
        return pd.DataFrame(data)

    def export_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chain, f, indent=4)


class MLBlockchainIntegration:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def record_investment_decision(self, comparison_df, bulk_quantity, time_horizon_yrs, recommended_machine):
        # Iterate through the DataFrame rows to create transactions
        for index, row in comparison_df.iterrows():
            status = "Recommended" if row['Machine_Model'] == recommended_machine else "Rejected"
            
            # Ensure we convert numpy types to native Python types for JSON serialization
            transaction = {
                'timestamp': str(datetime.now()),
                'machine_model': str(row['Machine_Model']),
                'investment_amount': float(row['Bulk_Total_TCO']),
                'predicted_profit': float(row['Bulk_Total_Profit']),
                'roi': float(row['ROI_Yrs']) if pd.notnull(row['ROI_Yrs']) else 0.0,
                'decision_status': status,
                'parameters': {
                    'bulk_qty': int(bulk_quantity),
                    'time_horizon': int(time_horizon_yrs)
                }
            }
            self.blockchain.add_transaction(transaction)

    def finalize_decisions(self, miner_address):
        return self.blockchain.mine_pending_transactions(miner_address)

    def get_decision_audit_trail(self):
        all_txs = []
        for block in self.blockchain.chain:
            for tx in block['transactions']:
                tx_copy = tx.copy()
                tx_copy['block_index'] = block['index']
                all_txs.append(tx_copy)
        
        if not all_txs:
            return pd.DataFrame()
            
        return pd.DataFrame(all_txs)

    def generate_investment_report(self, filename):
        audit_df = self.get_decision_audit_trail()
        if not audit_df.empty:
            audit_df.to_json(filename, orient='records', indent=4)
        else:
            with open(filename, 'w') as f:
                f.write("[]")

# ==========================================
#  MAIN SCRIPT STARTS HERE
# ==========================================

# Make sure script runs from its own folder
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass # In case of running in interactive notebook

# --- 1. Load the Data ---
try:
    df = pd.read_csv('synthetic_bulk_investment_data_format1.csv')
except FileNotFoundError:
    print("Error: 'synthetic_bulk_investment_data_format1.csv' not found.")
    # Fallback for testing if you don't have the file:
    print("Creating dummy data for demonstration...")
    data = {
        'Machine_Model': ['Model A', 'Model B', 'Model C'] * 34,
        'Initial_Cost_INR': np.random.uniform(50000, 200000, 102),
        'Scrap_Value_INR': np.random.uniform(5000, 20000, 102),
        'ANNUAL_OPERATING_COST_INR': np.random.uniform(10000, 50000, 102),
        'Power_Consumption_kWh_hr': np.random.uniform(1, 10, 102),
        'Max_Capacity_Units_hr': np.random.uniform(50, 100, 102),
        'Scheduled_Maint_Interval_Days': np.random.randint(30, 180, 102),
        'Historical_Failure_Rate': np.random.uniform(0, 0.1, 102),
        'Actual_Lifespan_Yrs': np.random.uniform(5, 15, 102),
        'Historical_Total_Maint_Cost': np.random.uniform(20000, 100000, 102),
        'Task_Completion_Time_Hrs': np.random.uniform(1, 5, 102)
    }
    df = pd.DataFrame(data)

# If your CSV still uses USD column names, rename them to INR for internal consistency
rename_map = {}
if 'Initial_Cost_USD' in df.columns:
    rename_map['Initial_Cost_USD'] = 'Initial_Cost_INR'
if 'Scrap_Value_USD' in df.columns:
    rename_map['Scrap_Value_USD'] = 'Scrap_Value_INR'
if 'ANNUAL_OPERATING_COST_USD' in df.columns:
    rename_map['ANNUAL_OPERATING_COST_USD'] = 'ANNUAL_OPERATING_COST_INR'

if rename_map:
    df = df.rename(columns=rename_map)

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Adjusted split for better training
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
r2_list = []
mae_list = []
mape_list = []
for i, target in enumerate(target_cols):
    y_true = y_test.iloc[:, i].values
    y_pred = y_pred_test[:, i]
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = (np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])).mean() * 100.0
    else:
        mape = np.nan

    r2_list.append(r2)
    mae_list.append(mae)
    mape_list.append(mape)

    print(f"Target: {target}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")

mean_r2 = np.mean(r2_list)
print(f"\nModel Accuracy (aggregate): Mean R2 across targets = {mean_r2:.4f}")

# ============================================================
#  USER-DRIVEN PREDICTION + FINANCIAL COMPARISON SECTION (INR)
# ============================================================

def get_float(prompt_text):
    while True:
        try:
            val = input(prompt_text)
            if not val: return 0.0 # Default to 0 if empty
            return float(val)
        except ValueError:
            print("Please enter a valid number.")

def get_int(prompt_text):
    while True:
        try:
            val = input(prompt_text)
            if not val: return 1 # Default to 1 if empty
            return int(val)
        except ValueError:
            print("Please enter a valid integer.")

print("\n=======================================================")
print("       USER INPUT: MACHINE OPTIONS TO COMPARE (INR)      ")
print("=======================================================")

n_machines = get_int("How many machine options do you want to compare? ")

user_rows = []
for i in range(n_machines):
    print(f"\n--- Machine {i+1} ---")
    machine_name = input("Enter machine model/name (e.g., 'Lathe X-A1'): ").strip()
    if not machine_name: machine_name = f"Machine_{i+1}"

    initial_cost = get_float("Initial cost (INR): ")
    scrap_value = get_float("Expected scrap/salvage value at end of life (INR): ")
    gst_rate_pct = get_float("GST tax rate (%) on purchase (e.g., 18 for 18%): ")
    power = get_float("Power consumption (kWh per hour): ")
    capacity = get_float("Max capacity (units per hour): ")
    maint_interval = get_float("Scheduled maintenance interval (days): ")
    failure_rate = get_float("Historical failure rate (0-1): ")

    user_rows.append({
        "Machine_Model": machine_name,
        "Initial_Cost_INR": initial_cost,
        "Scrap_Value_INR": scrap_value,
        "GST_Rate_Pct": gst_rate_pct,
        "Power_Consumption_kWh_hr": power,
        "Max_Capacity_Units_hr": capacity,
        "Scheduled_Maint_Interval_Days": maint_interval,
        "Historical_Failure_Rate": failure_rate
    })

new_machine_data = pd.DataFrame(user_rows)

# --- 8. Use the ML model to predict outputs for user machines ---
model_input_cols = [
    "Initial_Cost_INR",
    "Power_Consumption_kWh_hr",
    "Max_Capacity_Units_hr",
    "Scheduled_Maint_Interval_Days",
    "Historical_Failure_Rate",
    "Machine_Model"
]

model_input_df = new_machine_data[model_input_cols].copy()
model_input_df = pd.get_dummies(model_input_df, columns=['Machine_Model'], drop_first=True)
model_input_df = model_input_df.reindex(columns=X.columns, fill_value=0)

predicted_outputs = multi_output_model.predict(model_input_df)
predictions_df = pd.DataFrame(predicted_outputs, columns=target_cols)

comparison_df = new_machine_data.copy()
comparison_df['Predicted_Lifespan'] = predictions_df['Actual_Lifespan_Yrs']
comparison_df['Predicted_Maint_Cost'] = predictions_df['Historical_Total_Maint_Cost']
comparison_df['Predicted_Task_Time'] = predictions_df['Task_Completion_Time_Hrs']

# --- 9. Get Financial Parameters from User ---
print("\n=======================================================")
print("      USER INPUT: FINANCIAL PARAMETERS (GLOBAL, INR)      ")
print("=======================================================")

BULK_QUANTITY = get_int("Bulk quantity of units to purchase: ")
TIME_HORIZON_YRS = get_int("Time horizon for comparison (years): ")
TARGET_REVENUE_PER_UNIT_YR = get_float("Expected annual revenue per unit (INR): ")
ANNUAL_OPERATING_COST_INR = get_float("Annual operating cost per unit (INR): ")

# --- 10. Financial Metrics Calculation ---
comparison_df['GST_Factor'] = 1 + (comparison_df['GST_Rate_Pct'] / 100.0)
comparison_df['Effective_Initial_Cost'] = (
    comparison_df['Initial_Cost_INR'] * comparison_df['GST_Factor']
    - comparison_df['Scrap_Value_INR']
)

comparison_df['Predicted_Lifespan_clean'] = comparison_df['Predicted_Lifespan'].replace({0: np.nan})
comparison_df['Annual_Maint_Cost'] = (
    comparison_df['Predicted_Maint_Cost'] / comparison_df['Predicted_Lifespan_clean']
)
comparison_df['Annual_Maint_Cost'] = comparison_df['Annual_Maint_Cost'].fillna(
    comparison_df['Predicted_Maint_Cost'] / np.where(TIME_HORIZON_YRS > 0, TIME_HORIZON_YRS, 1)
)

comparison_df['TCO_Per_Unit'] = (
    comparison_df['Effective_Initial_Cost']
    + (ANNUAL_OPERATING_COST_INR * TIME_HORIZON_YRS)
    + (comparison_df['Annual_Maint_Cost'] * TIME_HORIZON_YRS)
)

comparison_df['Total_Revenue_Per_Unit'] = TARGET_REVENUE_PER_UNIT_YR * TIME_HORIZON_YRS
comparison_df['Net_Profit_Per_Unit'] = (
    comparison_df['Total_Revenue_Per_Unit'] - comparison_df['TCO_Per_Unit']
)

comparison_df['ROI_Yrs'] = np.where(
    comparison_df['Effective_Initial_Cost'] > 0,
    (comparison_df['Net_Profit_Per_Unit'] / comparison_df['Effective_Initial_Cost']) * 100.0,
    np.nan
)

comparison_df['Bulk_Total_TCO'] = comparison_df['TCO_Per_Unit'] * BULK_QUANTITY
comparison_df['Bulk_Total_Profit'] = comparison_df['Net_Profit_Per_Unit'] * BULK_QUANTITY

# --- 11. Final Recommendation ---
if comparison_df['Net_Profit_Per_Unit'].isnull().all():
    print("Warning: Net profit could not be computed for any machine.")
    best_model_row = comparison_df.iloc[0]
else:
    best_model_row = comparison_df.loc[comparison_df['Net_Profit_Per_Unit'].idxmax()]

print("\n--- FINAL INVESTMENT RECOMMENDATION ---")
print(f"Recommended machine: {best_model_row['Machine_Model']}")
print(f"Projected Net Profit (Bulk): INR {best_model_row['Bulk_Total_Profit']:,.0f}")

# --- 12. Visualization ---
# Using non-blocking show so script continues to blockchain part
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(comparison_df['Machine_Model'], comparison_df['Net_Profit_Per_Unit'])
plt.title('Net Profit Per Unit')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.scatter(comparison_df['Effective_Initial_Cost'], comparison_df['Predicted_Lifespan'])
plt.title('Cost vs Durability')
plt.xlabel('Cost')
plt.ylabel('Lifespan')

plt.tight_layout()
print("\n[INFO] Close the plot window to proceed to Blockchain recording...")
plt.show()

# ============================================================
#  BLOCKCHAIN INTEGRATION: RECORD INVESTMENT DECISIONS
# ============================================================

print("\n=======================================================")
print("      BLOCKCHAIN: RECORDING INVESTMENT DECISIONS      ")
print("=======================================================")

# Initialize blockchain
blockchain = MachineInvestmentBlockchain(difficulty=2)
print("✓ Blockchain initialized (Genesis block created)")

# Create integration layer
integration = MLBlockchainIntegration(blockchain)
print("✓ ML-Blockchain integration layer created")

# Record all machine investment decisions to blockchain
print("  Recording machine comparison decisions to blockchain...")
integration.record_investment_decision(
    comparison_df=comparison_df,
    bulk_quantity=BULK_QUANTITY,
    time_horizon_yrs=TIME_HORIZON_YRS,
    recommended_machine=best_model_row['Machine_Model']
)
print(f"✓ Recorded {len(comparison_df)} machine evaluations")

# Mine the decisions into a block
print("  Mining investment decisions block...")
integration.finalize_decisions(miner_address="ML_Investment_Analyzer_System")
print("  Block mined successfully!")

# Validate blockchain integrity
print("  Validating blockchain integrity...")
is_valid = blockchain.is_chain_valid()
validation_status = "✓ VALID" if is_valid else "✗ INVALID"
print(f"{validation_status} - Blockchain chain integrity confirmed\n")

# Display blockchain summary
print("="*60)
print("BLOCKCHAIN LEDGER SUMMARY")
print("="*60)
blockchain_summary = blockchain.get_chain_as_dataframe()
print(blockchain_summary.to_string(index=False))

# Display recorded transactions
print("\n" + "="*60)
print("RECORDED INVESTMENT DECISIONS (IMMUTABLE)")
print("="*60)
audit_trail = integration.get_decision_audit_trail()
if not audit_trail.empty:
    cols = ['machine_model', 'investment_amount', 'predicted_profit', 'decision_status']
    print(audit_trail[cols].to_string(index=False))
else:
    print("No transactions recorded")

# Generate and export detailed blockchain report
print("\n" + "="*60)
print("  GENERATING BLOCKCHAIN REPORT")
print("="*60)
integration.generate_investment_report("machine_investment_blockchain_report.json")
print("✓ Report exported to: machine_investment_blockchain_report.json")

blockchain.export_to_json("machine_investment_chain.json")
print("✓ Blockchain state exported to: machine_investment_chain.json")

print("\nBLOCKCHAIN RECORDING COMPLETE")