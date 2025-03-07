import streamlit as st
import pandas as pd
import joblib

# Load trained Isolation Forest model
model = joblib.load("isolation_forest_model (3).pkl")

# Streamlit UI
st.title("🔍 Insurance Fraud Detection")

st.write("Enter the details below to check if the claim is fraudulent.")

# Input fields for user data
months_as_customer = st.number_input("Months as Customer", min_value=1, max_value=600)
age = st.number_input("Age", min_value=18, max_value=100)
policy_deductable = st.number_input("Policy Deductible", min_value=0, max_value=5000)
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=100, max_value=50000)
umbrella_limit = st.number_input("Umbrella Limit", min_value=0, max_value=10000000)
total_claim_amount = st.number_input("Total Claim Amount", min_value=0, max_value=1000000)
injury_claim = st.number_input("Injury Claim", min_value=0, max_value=500000)
property_claim = st.number_input("Property Claim", min_value=0, max_value=500000)
vehicle_claim = st.number_input("Vehicle Claim", min_value=0, max_value=500000)

# Create DataFrame with input
user_data = pd.DataFrame([[months_as_customer, age, policy_deductable, 
                           policy_annual_premium, umbrella_limit, 
                           total_claim_amount, injury_claim, 
                           property_claim, vehicle_claim]],
                         columns=["months_as_customer", "age", "policy_deductable", 
                                  "policy_annual_premium", "umbrella_limit", 
                                  "total_claim_amount", "injury_claim", 
                                  "property_claim", "vehicle_claim"])

# Predict Fraud or Not
if st.button("Detect Fraud"):
    anomaly_score = model.decision_function(user_data)
    prediction = model.predict(user_data)

    # Display Results
    st.subheader("🔹 Prediction Result")
    if prediction[0] == -1:
        st.error("🚨 Fraudulent Claim Detected!")
    else:
        st.success("✅ The claim seems normal.")

    st.write(f"**Anomaly Score:** {anomaly_score[0]:.4f}")

