import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Preprocessors ---
# Ensure these files are in the same directory as your Streamlit app
try:
    best_model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    encoded_columns = joblib.load('encoded_columns.pkl')
    optimal_threshold = joblib.load('optimal_threshold.pkl')
except FileNotFoundError:
    st.error("Error: Model artifacts not found. Please ensure 'best_churn_model.pkl', 'scaler.pkl', 'selected_features.pkl', 'encoded_columns.pkl', and 'optimal_threshold.pkl' are in the same directory as this app.py file.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title='Telecom Churn Prediction', layout='centered')
st.title('📞 Telecom Customer Churn Prediction App')
st.markdown("Predict whether a customer will churn based on their service usage and demographic information.")

# --- Feature Input Section ---
st.header('Customer Information')

# Group inputs into columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.checkbox('Senior Citizen', value=False)
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

with col2:
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

with col3:
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    tenure = st.slider('Tenure (months)', 0, 72, 12) # min=0 to allow new customers
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=120.0, value=70.0, step=0.5)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=9000.0, value=400.0, step=10.0)

# --- Prediction Logic ---
if st.button('Predict Churn'):
    # Create a dictionary from inputs
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # --- Replicate Feature Engineering (EXACTLY as in notebook) ---
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df.loc[input_df['tenure'] == 0, 'TotalCharges'] = 0
    input_df['PaperlessBilling'] = input_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    input_df['Partner'] = input_df['Partner'].map({'Yes': 1, 'No': 0})
    input_df['Dependents'] = input_df['Dependents'].map({'Yes': 1, 'No': 0})

    # Tenure-based Features
    input_df['TenureGroup'] = pd.cut(input_df['tenure'], bins=[0, 12, 24, 48, np.inf], labels=['New', 'Regular', 'Established', 'Loyal'], right=False)
    input_df['IsFirstYear'] = (input_df['tenure'] < 12).astype(int) # Change from <=12 to <12 as per bin edge
    input_df['IsLongTerm'] = (input_df['tenure'] >= 24).astype(int)

    # Charge-based Features
    # Handle division by zero for new customers (tenure=0)
    input_df['AvgMonthlyCharge'] = input_df.apply(lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1)
    input_df['CustomerLTV'] = input_df['TotalCharges'] + (input_df['MonthlyCharges'] * 6)

    # Service-based Features
    additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['NumAdditionalServices'] = input_df[additional_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    input_df['HasSecurityBundle'] = ((input_df['OnlineSecurity'] == 'Yes') & (input_df['OnlineBackup'] == 'Yes')).astype(int)
    input_df['HasStreamingBundle'] = ((input_df['StreamingTV'] == 'Yes') & (input_df['StreamingMovies'] == 'Yes')).astype(int)
    input_df['InternetUser'] = (input_df['InternetService'] != 'No').astype(int)
    input_df['FiberOpticUser'] = (input_df['InternetService'] == 'Fiber optic').astype(int)
    input_df['ServicesPerMonth'] = input_df['NumAdditionalServices'] / (input_df['tenure'] + 1)

    # Contract / Payment Features
    input_df['IsMonthToMonth'] = (input_df['Contract'] == 'Month-to-month').astype(int)
    input_df['ContractType'] = input_df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    input_df['ElectronicPayment'] = input_df['PaymentMethod'].str.contains('electronic check|automatic', case=False).astype(int)
    input_df['PaymentRisk'] = input_df['PaymentMethod'].map({
        'Electronic check': 3, 'Mailed check': 2,
        'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1
    })
    input_df['PaperlessHighRisk'] = ((input_df['PaperlessBilling'] == 1) & (input_df['PaymentMethod'] == 'Electronic check')).astype(int)

    # Family / Demographic Features
    input_df['HasFamily'] = ((input_df['Partner'] == 1) | (input_df['Dependents'] == 1)).astype(int)

    # Risk / Engagement Features
    # Use the median monthly charges from the training data (70.35) for consistency
    input_df['HighCostLowTenure'] = (
        (input_df['MonthlyCharges'] > 70.35) & (input_df['tenure'] < 12)
    ).astype(int)
    # Engagement Score - max tenure is 72 from training data
    input_df['EngagementScore'] = (
        input_df['NumAdditionalServices'] * 0.3 +
        input_df['ContractType'] * 0.4 +
        (input_df['tenure'] / 72) * 0.3
    )

    # --- Encoding and Scaling ---
    # One-hot encode categorical columns, ensuring all columns from training are present
    processed_df = pd.get_dummies(input_df)
    # Align columns to match the training data, filling missing ones with 0
    processed_df = processed_df.reindex(columns=encoded_columns, fill_value=0)

    # Scale numerical features
    processed_scaled = scaler.transform(processed_df)

    # --- Feature Selection ---
    # Create a mask for features selected during training
    feature_mask_indices = [encoded_columns.get_loc(f) for f in selected_features]
    final_input = processed_scaled[:, feature_mask_indices]

    # Make prediction
    churn_probability = best_model.predict_proba(final_input)[:, 1][0]
    churn_prediction = (churn_probability >= optimal_threshold).astype(int)

    st.subheader('Prediction Results:')
    
    # Display churn probability with a metric and progress bar
    st.metric(label="Predicted Churn Probability", value=f"{churn_probability:.2%}", delta_color="off")
    st.progress(churn_probability, text=f"Probability of Churn ({churn_probability:.2%})")

    if churn_prediction == 1:
        st.error(f'This customer is predicted to **CHURN** (using optimal threshold of {optimal_threshold:.2f})')
        st.warning('Consider proactive retention strategies for this customer.')
    else:
        st.success(f'This customer is predicted **NOT to churn** (using optimal threshold of {optimal_threshold:.2f})')
        st.info('This customer is likely stable.')
