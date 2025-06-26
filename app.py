import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("xgb_fraud_model.pkl")

# Define expected feature names in correct order
feature_names = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
                 'errorBalanceOrig', 'errorBalanceDest']

st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #141e30, #243b55);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #ff4b5c;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
        }
        .stTextInput>div>input, .stSelectbox>div>div, .stNumberInput>div>input {
            background-color: #1e2a38;
            color: white;
            border-radius: 5px;
            font-weight: 500;
        }
        .stMarkdown h3 {
            color: #00d8ff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Real-Time Fraud Detection")
st.markdown("### 🧾 Enter Transaction Details Below")

# Input fields
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(" Amount", min_value=0.0, step=500.0, format="%.2f")
        oldbalanceOrg = st.number_input(" Sender Old Balance", min_value=0.0, step=500.0, format="%.2f")
        oldbalanceDest = st.number_input(" Receiver Old Balance", min_value=0.0, step=500.0, format="%.2f")
    with col2:
        newbalanceOrig = st.number_input(" Sender New Balance", min_value=0.0, step=500.0, format="%.2f")
        newbalanceDest = st.number_input(" Receiver New Balance", min_value=0.0, step=500.0, format="%.2f")
        transaction_type = st.selectbox(" Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    submitted = st.form_submit_button(" Predict Transaction")

    if submitted:
        # Derived features
        errorBalanceOrig = oldbalanceOrg - newbalanceOrig - amount
        errorBalanceDest = newbalanceDest - oldbalanceDest - amount

        # One-hot encoding manually
        type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
        type_DEBIT = 1 if transaction_type == "DEBIT" else 0
        type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
        type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

        input_data = pd.DataFrame([{
            'step': 1,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_CASH_OUT': type_CASH_OUT,
            'type_DEBIT': type_DEBIT,
            'type_PAYMENT': type_PAYMENT,
            'type_TRANSFER': type_TRANSFER,
            'errorBalanceOrig': errorBalanceOrig,
            'errorBalanceDest': errorBalanceDest,
        }])[feature_names]  # Ensure correct feature order

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")
        st.markdown("##  Prediction Result")

        result_container = st.container()
        with result_container:
            if prediction == 1:
                st.markdown(f"""
                    <div style="background-color:#FF4B4B;padding:20px;border-radius:10px">
                        <h2 style="color:white;"> Fraud Detected!</h2>
                        <p style="color:white;font-size:18px;">
                            This transaction is flagged as fraudulent.<br>
                            <strong>Confidence:</strong> {probability:.2f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color:#2ECC71;padding:20px;border-radius:10px">
                        <h2 style="color:white;"> Legitimate Transaction</h2>
                        <p style="color:white;font-size:18px;">
                            This transaction is considered safe.<br>
                            <strong>Confidence:</strong> {100 - probability:.2f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)

st.markdown("###  Project Summary and Insights")
with st.expander(" Click to View Findings, Risks & Improvements"):
    st.markdown("""
    ** Key Findings:**
    - The model achieves **~99.97% accuracy** and **AUC 0.9997** on test data.
    - High recall (0.98) ensures **fraudulent activities are rarely missed**.

    ** Risks:**
    - May produce **false positives**, flagging legitimate users occasionally.
    - Sensitive to **data imbalance**; fraud data was rare and needed SMOTE.

   **🔧 Future Improvements:**
    -  **Live Integration**: Connect to a real-time API or streaming service for instant transaction analysis.
    -  **Self-Learning**: Add a feedback loop so the model can learn continuously from new data.
    -  **Access Control**: Enable role-based access and restrict sensitive information to authorized users.
    -  **Audit Trails**: Maintain a full log of predictions and changes for security and traceability.
    -  **Model Explainability**: Integrate SHAP or LIME to explain why a transaction is flagged.
    -  **Monitoring Tools**: Add real-time monitoring and alerting for model performance.
    -  **Reduce False Alarms**: Fine-tune thresholds and precision to minimize disruptions.
    -  **Deploy Anywhere**: Containerize the model with Docker for easy deployment across environments.
    -  **Feature Expansion**: Include behavioral, geographic, or device-related features for richer insights.
    -  **Fraud Alerts**: Notify fraud teams via email or messaging platforms in real-time.
    -  **Version Control**: Use MLflow or DVC to track changes, experiments, and rollbacks.
    -  **Support Integration**: Link flagged transactions with CRM tools to support follow-up actions.
    -  **Privacy Compliance**: Ensure strong data protection and GDPR/PII compliance practices.
    """)
