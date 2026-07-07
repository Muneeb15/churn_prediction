import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and column structure
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction & Retention Insights")

tab1, tab2 = st.tabs(["🔮 Predict Churn", "📈 Business Insights"])

# ---------------- TAB 1: PREDICTION ----------------
with tab1:
    st.header("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col3:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    if st.button("Predict Churn Risk", type="primary"):
        # Build input row matching training structure
        input_dict = {col: 0 for col in model_columns}

        # Fill in the values (manual mapping matching your encoding)
        input_dict['Gender'] = 1 if gender == "Male" else 0
        input_dict['Senior Citizen'] = 1 if senior == "Yes" else 0
        input_dict['Partner'] = 1 if partner == "Yes" else 0
        input_dict['Dependents'] = 1 if dependents == "Yes" else 0
        input_dict['Phone Service'] = 1 if phone_service == "Yes" else 0
        input_dict['Paperless Billing'] = 1 if paperless == "Yes" else 0
        input_dict['Tenure Months'] = tenure
        input_dict['Monthly Charges'] = monthly_charges
        input_dict['Total Charges'] = total_charges

        # Contract risk score
        contract_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        input_dict['Contract Risk Score'] = contract_map[contract]

        # One-hot columns - set to 1 if they exist in model_columns
        if internet == "Fiber optic" and 'Internet Service_Fiber optic' in input_dict:
            input_dict['Internet Service_Fiber optic'] = 1
        if internet == "No" and 'Internet Service_No' in input_dict:
            input_dict['Internet Service_No'] = 1

        payment_col = f'Payment Method_{payment}'
        if payment_col in input_dict:
            input_dict[payment_col] = 1

        contract_col = f'Contract_{contract}'
        if contract_col in input_dict:
            input_dict[contract_col] = 1

        # Convert to DataFrame in correct column order
        input_df = pd.DataFrame([input_dict])[model_columns]

        # Scale numeric columns the same way as training
        numerical_columns = ['Tenure Months', 'Monthly Charges', 'Total Charges']
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK — Probability: {probability:.1%}")
        else:
            st.success(f"✅ LOW CHURN RISK — Probability: {probability:.1%}")

# ---------------- TAB 2: BUSINESS INSIGHTS ----------------
with tab2:
    st.header("Key Churn Drivers")
    st.markdown("""
    Based on our trained model, the top factors driving customer churn are:

    1. **Total Charges** — cumulative spend is the strongest churn signal
    2. **Monthly Charges** — higher bills correlate with higher churn
    3. **Tenure** — new customers (0-12 months) are highest risk
    4. **Contract Type** — month-to-month customers churn far more than annual contracts
    5. **Service Count** — customers with fewer add-on services churn more
    6. **Fiber Optic Internet** — this segment shows elevated churn risk
    7. **Payment Method** — electronic check users churn more than autopay users
    """)

    st.header("Retention Recommendations")
    st.markdown("""
    - 🎯 Offer contract upgrade incentives to month-to-month customers
    - 🎯 Build a strong onboarding program for the first 3 months
    - 🎯 Bundle add-on services (Tech Support, Online Security) at a discount
    - 🎯 Investigate fiber optic pricing/service quality
    - 🎯 Encourage autopay adoption with small incentives
    """)