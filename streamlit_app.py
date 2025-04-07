import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import math

model = load('loan_prd_model.joblib')
scaler = load('scaler.joblib')
la_model = load('loan_amount_model.joblib')

columns_order= ['income_annum',
                    'loan_amount',
                    'loan_term',
                    'cibil_score',
                    'residential_assets_value',
                    'commercial_assets_value',
                    'luxury_assets_value',
                    'bank_asset_value',
                    'loan_status']

# Define the columns that are numerical
numerical_cols = ['income_annum', 
                  'loan_amount', 
                  'loan_term', 
                  'cibil_score', 
                  'residential_assets_value', 
                  'commercial_assets_value', 
                  'luxury_assets_value',
                  'bank_asset_value']

# Title of the app
st.title("Loan Prediction Form")

# Collect user inputs
st.subheader("Enter your details:")

# Education status
education = st.selectbox("Education", ["Graduate", "Not Graduate"])

# Self-employed status
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

# Income per annum
income_annum = st.number_input("Income per annum", min_value=0, value=50000, step=1000)

# Loan amount
loan_amount = st.number_input("Loan Amount", min_value=0, value=500000, step=10000)

# Loan term
loan_term = st.number_input("Loan Term (in years)", min_value=1, value=10, step=1)

# CIBIL score
cibil_score = st.number_input("CIBIL Score", min_value=0, value=650, step=10)

# Residential assets value
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=1000000, step=10000)

# Commercial assets value
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=500000, step=10000)

# Luxury assets value
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=200000, step=10000)

# Bank asset value
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=300000, step=10000)

# Collect the input data into a list or dictionary
input_data = {
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value,
}

# Function to transform data into model's expected format
def transform_data(input_data):   

    transformed_data = {
        "income_annum": input_data["income_annum"],
        "loan_amount": input_data["loan_amount"],
        "loan_term": input_data["loan_term"],
        "cibil_score": input_data["cibil_score"],         
        "residential_assets_value": input_data["residential_assets_value"],
        "commercial_assets_value": input_data["commercial_assets_value"],
        "luxury_assets_value": input_data["luxury_assets_value"],
        "bank_asset_value": input_data["bank_asset_value"]    
    }
    
    
    # Convert the sample data into a DataFrame
    x_data = pd.DataFrame([transformed_data])
    
    x_data[numerical_cols] = scaler.transform(x_data[numerical_cols])

    return x_data

# When the user clicks the Submit button, pass the data to the model
if st.button("Submit"):
    # Transform data for model input
    transformed_input = transform_data(input_data)
    
    # Predict the loan status using the model
    loan_status = model.predict(transformed_input)  # Extract the first prediction from the result

    ya = transformed_input['loan_amount']
    Xa = transformed_input[['income_annum', 'loan_term', 'residential_assets_value', 
             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]

    # Predict the loan amount using the model
    loan_amount = la_model.predict(Xa)  # Extract the first prediction from the result
    print(type(ya))  # Should print <class 'float'> or <class 'int'>
    print(type(loan_amount))  # Should print <class 'float'> or <class 'int'>
    
    ya_reshaped = np.full((1, 8), ya)  # Create a row with 8 identical values
    actual_loan_amount = scaler.inverse_transform(ya_reshaped)[0, 1]
    loan_amount_reshaped = np.full((1, 8), loan_amount)  # Create a row with 8 identical values
    predicted_loan_amount = scaler.inverse_transform(loan_amount_reshaped)[0, 1]
    rounded_loan_amount = math.floor(predicted_loan_amount / 1000) * 1000
    #st.write(f"### Predicted Loan Amount:", predicted_loan_amount, actual_loan_amount)
    # Display result
    if loan_status == 1:
        if rounded_loan_amount >= actual_loan_amount:
            st.success("Loan Approved") 
        else:
            st.error("Loan Rejected")
            st.warning("The requested loan amount exceeds the predicted loan amount.")
            # Display the predicted loan amount
            st.write(f"### Predicted Loan Amount: {rounded_loan_amount:,.2f}")
    else:
        st.error("Loan Rejected")
        
    #st.write(f"### Loan Status: {'Approved' if loan_status == 1 else 'Rejected'}")

