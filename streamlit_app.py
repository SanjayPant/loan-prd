import streamlit as st
from joblib import load

model = load('final_model.joblib')

# Predict on new data
#prediction = final_model.predict(new_data)
# Title of the app
st.title("Loan Prediction Form")

# Collect user inputs
st.subheader("Enter your details:")

# Number of dependents
no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0, step=1)

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
    "no_of_dependents": no_of_dependents,
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

# Display input values
st.write("### Input Data Summary")
for key, value in input_data.items():
    st.write(f"{key.replace('_', ' ').title()}: {value}")

# Load your trained model (make sure the model file is in the same directory or give the correct path)
# model = mymodel.load()

# Function to transform data into model's expected format (this depends on your model's input structure)
def transform_data(input_data):
    # Example transformation (you may need to encode categorical variables or scale numerical ones)
    # This step needs to align with how your model expects the data
    transformed_data = [
        input_data["no_of_dependents"],
        1 if input_data["education"] == "Graduate" else 0,  # Encode "Graduate" as 1 and "Not Graduate" as 0
        1 if input_data["self_employed"] == "Yes" else 0,   # Encode "Yes" as 1 and "No" as 0
        input_data["income_annum"],
        input_data["loan_amount"],
        input_data["loan_term"],
        input_data["cibil_score"],
        input_data["residential_assets_value"],
        input_data["commercial_assets_value"],
        input_data["luxury_assets_value"],
        input_data["bank_asset_value"],
    ]
    return transformed_data

# When the user clicks the Submit button, pass the data to the model
if st.button("Submit"):
    # Transform data for model input
    transformed_input = transform_data(input_data)
    
    # Predict the loan status using the model (replace this with your actual prediction call)
    # Example: 
    loan_status = model.predict([transformed_input])
    loan_status = "Approved" if transformed_input[6] > 700 else "Rejected"  # Placeholder logic for demonstration

    # Display result
    st.write(f"### Loan Status: {loan_status}")
