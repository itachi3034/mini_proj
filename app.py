import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
import time
from datetime import datetime

# ==========================================
#  1. BLOCKCHAIN CLASSES (Logic Layer)
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
        new_block['hash'] = self.proof_of_work(new_block)
        self.chain.append(new_block)
        self.pending_transactions = []
        return new_block

    def proof_of_work(self, block):
        block['nonce'] = 0
        computed_hash = self.hash_block(block)
        while not computed_hash.startswith('0' * self.difficulty):
            block['nonce'] += 1
            computed_hash = self.hash_block(block)
        return computed_hash

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

class MLBlockchainIntegration:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def record_investment_decision(self, comparison_df, bulk_quantity, time_horizon_yrs, recommended_machine):
        count = 0
        for index, row in comparison_df.iterrows():
            status = "Recommended" if row['Machine_Model'] == recommended_machine else "Rejected"
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
            count += 1
        return count

    def finalize_decisions(self, miner_address):
        return self.blockchain.mine_pending_transactions(miner_address)
    
    def get_audit_trail(self):
        all_txs = []
        for block in self.blockchain.chain:
            for tx in block['transactions']:
                tx_copy = tx.copy()
                tx_copy['block_index'] = block['index']
                all_txs.append(tx_copy)
        return pd.DataFrame(all_txs)

# ==========================================
#  2. WEB APP CONFIG & SESSION STATE
# ==========================================

st.set_page_config(page_title="AI Investment Blockchain", layout="wide", page_icon="🔗")

# Initialize Blockchain in Session State (so it survives button clicks)
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = MachineInvestmentBlockchain(difficulty=3)
    st.session_state.integration = MLBlockchainIntegration(st.session_state.blockchain)

# ==========================================
#  3. MODEL TRAINING (Cached)
# ==========================================

@st.cache_resource
def train_model():
    # Load or Create Data
    try:
        df = pd.read_csv('synthetic_bulk_investment_data_format1.csv')
        if 'Initial_Cost_USD' in df.columns: df = df.rename(columns={'Initial_Cost_USD': 'Initial_Cost_INR'})
    except FileNotFoundError:
        # Create dummy data if file missing
        data = {
            'Machine_Model': ['Model A', 'Model B', 'Model C'] * 100,
            'Initial_Cost_INR': np.random.uniform(50000, 200000, 300),
            'Scrap_Value_INR': np.random.uniform(5000, 20000, 300),
            'ANNUAL_OPERATING_COST_INR': np.random.uniform(10000, 50000, 300),
            'Power_Consumption_kWh_hr': np.random.uniform(1, 10, 300),
            'Max_Capacity_Units_hr': np.random.uniform(50, 100, 300),
            'Scheduled_Maint_Interval_Days': np.random.randint(30, 180, 300),
            'Historical_Failure_Rate': np.random.uniform(0, 0.1, 300),
            'Actual_Lifespan_Yrs': np.random.uniform(5, 15, 300),
            'Historical_Total_Maint_Cost': np.random.uniform(20000, 100000, 300),
            'Task_Completion_Time_Hrs': np.random.uniform(1, 5, 300)
        }
        df = pd.DataFrame(data)

    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=['Machine_Model'], drop_first=True)
    target_cols = ['Actual_Lifespan_Yrs', 'Historical_Total_Maint_Cost', 'Task_Completion_Time_Hrs']
    
    X = df_encoded.drop(columns=target_cols)
    y = df_encoded[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CRITICAL FIX: n_jobs=1 prevents Windows crash and "ScriptRunContext" errors
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
    model.fit(X_train, y_train)
    
    return model, X.columns, df 

# Train the model once
model, model_features, raw_df = train_model()

# ==========================================
#  4. SIDEBAR - SETTINGS
# ==========================================

st.sidebar.title("⚙️ Financial Settings")
st.sidebar.markdown("Define your global investment parameters.")

BULK_QTY = st.sidebar.number_input("Bulk Quantity", min_value=1, value=10)
TIME_HORIZON = st.sidebar.number_input("Time Horizon (Years)", min_value=1, value=5)
REV_PER_UNIT = st.sidebar.number_input("Revenue / Unit / Year (INR)", min_value=0.0, value=20000.0)
OPS_COST = st.sidebar.number_input("Ops Cost / Unit / Year (INR)", min_value=0.0, value=5000.0)

st.sidebar.divider()
st.sidebar.info("Model Status: ✅ Trained")

# ==========================================
#  5. MAIN PAGE - INTERFACE
# ==========================================

st.title("🏭 AI & Blockchain Investment Analyzer")
st.markdown("Use AI to predict machine performance, calculate ROI, and immutably record decisions on a local Blockchain.")

# --- TABS FOR WORKFLOW ---
tab1, tab2, tab3 = st.tabs(["1. 📝 Input & Analysis", "2. 📊 Results & Charts", "3. 🔗 Blockchain Ledger"])

with tab1:
    st.subheader("Comparison Catalog")
    st.write("Edit the table below to add machines you want to compare. You can copy-paste from Excel or type directly.")

    # Default Data for the Editor
    default_catalog = pd.DataFrame([
        {"Machine_Model": "Heavy Duty X1", "Initial_Cost_INR": 150000, "Scrap_Value_INR": 15000, "GST_Rate_Pct": 18, "Power_Consumption_kWh_hr": 8.5, "Max_Capacity_Units_hr": 120, "Scheduled_Maint_Interval_Days": 90, "Historical_Failure_Rate": 0.02},
        {"Machine_Model": "Eco Runner V2", "Initial_Cost_INR": 85000, "Scrap_Value_INR": 8000, "GST_Rate_Pct": 18, "Power_Consumption_kWh_hr": 4.2, "Max_Capacity_Units_hr": 80, "Scheduled_Maint_Interval_Days": 60, "Historical_Failure_Rate": 0.05},
        {"Machine_Model": "Precision Pro Z", "Initial_Cost_INR": 210000, "Scrap_Value_INR": 25000, "GST_Rate_Pct": 28, "Power_Consumption_kWh_hr": 6.0, "Max_Capacity_Units_hr": 150, "Scheduled_Maint_Interval_Days": 120, "Historical_Failure_Rate": 0.01}
    ])

    # Interactive Data Editor - UPDATED FIX: width="stretch" (Fixes Screenshot 1)
    user_input_df = st.data_editor(default_catalog, num_rows="dynamic", width="stretch")

    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("AI is predicting machine lifespans and maintenance costs..."):
            # 1. Prepare Input for Model
            model_input = user_input_df[["Initial_Cost_INR", "Power_Consumption_kWh_hr", 
                                        "Max_Capacity_Units_hr", "Scheduled_Maint_Interval_Days", 
                                        "Historical_Failure_Rate", "Machine_Model"]].copy()
            
            # Encode and Align Columns
            model_input = pd.get_dummies(model_input, columns=['Machine_Model'], drop_first=True)
            model_input = model_input.reindex(columns=model_features, fill_value=0)
            
            # 2. Predict
            preds = model.predict(model_input)
            pred_df = pd.DataFrame(preds, columns=['Actual_Lifespan_Yrs', 'Historical_Total_Maint_Cost', 'Task_Completion_Time_Hrs'])
            
            # 3. Financial Math
            results = user_input_df.copy()
            results['Predicted_Lifespan'] = pred_df['Actual_Lifespan_Yrs']
            results['Predicted_Maint_Cost'] = pred_df['Historical_Total_Maint_Cost']
            
            # ROI Calculations
            results['GST_Factor'] = 1 + (results['GST_Rate_Pct'] / 100.0)
            results['Effective_Initial_Cost'] = (results['Initial_Cost_INR'] * results['GST_Factor'] - results['Scrap_Value_INR'])
            
            # Maint Cost Annualized
            results['Annual_Maint_Cost'] = results['Predicted_Maint_Cost'] / results['Predicted_Lifespan'].replace({0: np.nan})
            results['Annual_Maint_Cost'] = results['Annual_Maint_Cost'].fillna(0)
            
            # TCO & Profit
            results['TCO_Per_Unit'] = (
                results['Effective_Initial_Cost'] + 
                (OPS_COST * TIME_HORIZON) + 
                (results['Annual_Maint_Cost'] * TIME_HORIZON)
            )
            
            results['Total_Revenue_Per_Unit'] = REV_PER_UNIT * TIME_HORIZON
            results['Net_Profit_Per_Unit'] = results['Total_Revenue_Per_Unit'] - results['TCO_Per_Unit']
            results['ROI_Yrs'] = (results['Net_Profit_Per_Unit'] / results['Effective_Initial_Cost']) * 100.0
            
            results['Bulk_Total_TCO'] = results['TCO_Per_Unit'] * BULK_QTY
            results['Bulk_Total_Profit'] = results['Net_Profit_Per_Unit'] * BULK_QTY

            # Save results to session state to pass to other tabs
            st.session_state.results = results
            st.success("Analysis Complete! Go to the 'Results' tab.")

with tab2:
    if 'results' in st.session_state:
        results = st.session_state.results
        best_machine = results.loc[results['Net_Profit_Per_Unit'].idxmax()]
        
        # Metric Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Best Choice", best_machine['Machine_Model'])
        col2.metric("Projected Bulk Profit", f"₹ {best_machine['Bulk_Total_Profit']:,.0f}")
        col3.metric("ROI", f"{best_machine['ROI_Yrs']:.1f}%")
        
        st.subheader("Detailed Financial Breakdown")
        st.dataframe(results[['Machine_Model', 'Initial_Cost_INR', 'Predicted_Lifespan', 'Net_Profit_Per_Unit', 'ROI_Yrs']].style.format({
            'Initial_Cost_INR': '₹{:.2f}', 
            'Predicted_Lifespan': '{:.1f} Yrs',
            'Net_Profit_Per_Unit': '₹{:.2f}',
            'ROI_Yrs': '{:.1f}%'
        }))
        
        # Charts
        st.subheader("Profitability Comparison")
        st.bar_chart(results, x="Machine_Model", y="Net_Profit_Per_Unit", color="#4CAF50")
        
        st.subheader("Cost vs Durability")
        fig, ax = plt.subplots()
        ax.scatter(results['Effective_Initial_Cost'], results['Predicted_Lifespan'])
        for i, txt in enumerate(results['Machine_Model']):
            ax.annotate(txt, (results['Effective_Initial_Cost'].iloc[i], results['Predicted_Lifespan'].iloc[i]))
        ax.set_xlabel("Effective Cost")
        ax.set_ylabel("Predicted Lifespan (Yrs)")
        st.pyplot(fig)
        
        st.divider()
        if st.button("🔒 Record Decision to Blockchain"):
            count = st.session_state.integration.record_investment_decision(
                results, BULK_QTY, TIME_HORIZON, best_machine['Machine_Model']
            )
            st.session_state.integration.finalize_decisions("Admin_User")
            st.toast(f"Success! {count} records mined to block.", icon="🔗")
            
    else:
        st.info("Please run the analysis in Tab 1 first.")

with tab3:
    st.subheader("🔗 Live Blockchain Ledger")
    st.write("This ledger is immutable. Once data is added here, it cannot be deleted.")
    
    # 1. Chain View - UPDATED FIX: width="stretch" (Fixes Screenshot 1)
    chain_df = st.session_state.blockchain.get_chain_as_dataframe()
    st.dataframe(chain_df, width="stretch")
    
    # 2. Transaction View - UPDATED FIX: width="stretch" (Fixes Screenshot 1)
    st.subheader("📜 Audit Trail (Transactions)")
    audit_df = st.session_state.integration.get_audit_trail()
    
    if not audit_df.empty:
        display_cols = ['timestamp', 'block_index', 'machine_model', 'predicted_profit', 'decision_status']
        st.dataframe(audit_df[display_cols], width="stretch")
        
        # Download Button
        json_data = audit_df.to_json(orient='records', indent=4)
        st.download_button("Download Audit Report (JSON)", json_data, "blockchain_audit.json", "application/json")
    else:
        st.warning("No transactions recorded yet.")