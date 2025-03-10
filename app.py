# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from database import init_db, register_user, login_user, get_all_users

# Initialize the database
init_db()

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Function to preprocess input data (assuming you already have this function)
def preprocess_input_data(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    Gender = 1 if Gender == 'Male' else 0
    Married = 1 if Married == 'Yes' else 0
    Education = 1 if Education == 'Graduate' else 0
    Self_Employed = 1 if Self_Employed == 'Yes' else 0
    Property_Area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[Property_Area]
    ApplicantIncomelog = np.log(ApplicantIncome) if ApplicantIncome > 0 else None
    LoanAmountLog = np.log(LoanAmount) if LoanAmount > 0 else None
    Loan_Amount_Term_log = np.log(Loan_Amount_Term) if Loan_Amount_Term > 0 else None

    data = {    
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncomelog': [ApplicantIncomelog],
        'LoanAmountLog': [LoanAmountLog],
        'Loan_Amount_Term_log': [Loan_Amount_Term_log],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    }
    return pd.DataFrame(data)[model.feature_names_in_]

# Initialize session state for login tracking and page navigation
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"  # Options: "login", "register", "main"

# Navigation function to set the target page
def navigate_to(page):
    st.session_state["page"] = page

# Render Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password and login_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Logged in successfully!")
            navigate_to("main")  # Redirect to main page after login
        else:
            st.error("Invalid credentials or empty fields.")
    if st.button("Go to Register"):
        navigate_to("register")

# Render Registration Page with validation
def register_page():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if not username or not password:
            st.error("Please fill in all fields.")
        elif register_user(username, password):
            st.success("User registered successfully! Please log in.")
            navigate_to("login")
        else:
            st.error("Username already taken.")
    if st.button("Go to Login"):
        navigate_to("login")

# Render Main Prediction Page (Only for logged-in users)
def main_page():
    st.title("Loan Approval Prediction App")
    st.write(f"Welcome, {st.session_state['username']}!")
    
    # Sidebar input for loan prediction parameters
    st.sidebar.header('User Input Parameters')
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    Education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    ApplicantIncome = st.sidebar.number_input('Applicant Income', min_value=0, help="Enter a positive number")
    LoanAmount = st.sidebar.number_input('Loan Amount', min_value=0, help="Enter a positive number")
    Loan_Amount_Term = st.sidebar.number_input('Loan Amount Term', min_value=0, help="Enter a positive number")
    Credit_History = st.sidebar.selectbox('Credit History', (1.0, 0.0))
    Property_Area = st.sidebar.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    # Prediction and validation
    if st.sidebar.button("Predict"):
        # Basic validation for non-zero numeric values
        if ApplicantIncome <= 0:
            st.error("Applicant Income must be greater than zero.")
        elif LoanAmount <= 0:
            st.error("Loan Amount must be greater than zero.")
        elif Loan_Amount_Term <= 0:
            st.error("Loan Amount Term must be greater than zero.")
        else:
            # Preprocess and predict if inputs are valid
            input_df = preprocess_input_data(
                Gender, Married, Dependents, Education, Self_Employed, 
                ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
            )
            prediction = model.predict(input_df)

            # Display prediction result
            if prediction[0] == 1:
                st.write("Loan Approved ✅")
            else:
                st.write("Loan Denied ❌")

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        navigate_to("login")

# Display pages based on login status and page navigation
if st.session_state["logged_in"]:
    if st.session_state["page"] == "main":
        main_page()
else:
    if st.session_state["page"] == "login":
        login_page()
    elif st.session_state["page"] == "register":
        register_page()
