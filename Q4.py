import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import re

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Loading the trained model and scaler
model, scaler = None, None
try:
    model = joblib.load("heart_disease_model.joblib")
except Exception as e:
    st.error(f"Error loading the model: {e}")

try:
    # Load the scaler trained with the correct features
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"Error loading the scaler: {e}")

# Checking if both the model and scaler are loaded successfully
if model and scaler:
    # Creating a form to input patient details
    form = st.form("patient_details")
    patient_details = {}
    for column in columns:
        # Adding thw input fields for each feature
        if column == 'sex':
            patient_details[column] = form.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
        elif column == 'cp':
            patient_details[column] = form.selectbox(
                'cp (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal Pain, 4 = Asymptomatic)', [1, 2, 3, 4])

        elif column == 'fbs':
            patient_details[column] = form.selectbox('fbs (> 120 mg/dl) (0 = False, 1 = True)', [0, 1])
        elif column == 'exang':
            patient_details[column] = form.selectbox('exang (0 = No, 1 = Yes)', [0, 1])
        elif column == 'restecg':
            patient_details[column] = form.selectbox(
                'restecg (0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy)', [0, 1, 2])
        elif column == 'slope':
            patient_details[column] = form.selectbox('slope (0 = Upsloping, 1 = Flat, 2 = Downsloping)', [0, 1, 2])
        elif column == 'ca':
            patient_details[column] = form.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
        elif column == 'thal':
            patient_details[column] = form.selectbox('thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)',
                                                     [0, 1, 2])
        elif column == 'oldpeak':
            patient_details[column] = form.number_input('Oldpeak', min_value=0.0, max_value=10.0, step=0.1,
                                                        format="%.1f")
        else:
            patient_details[column] = form.number_input(column.capitalize(), min_value=0, max_value=500, step=1)

    submit = form.form_submit_button("Submit")

    if submit:
        # Processing the input data
        try:
            # Validating input
            validation_passed = True
            for col, val in patient_details.items():
                if not str(val).isnumeric():
                    st.error(f"Invalid value for {col.capitalize()}. Please enter a numeric value.")
                    validation_passed = False
                    break
                elif not isinstance(val, int):
                    st.error(f"Invalid value for {col.capitalize()}. Please enter an integer value.")
                    validation_passed = False
                    break

            if validation_passed:
                # Converting values to appropriate data types if needed
                # Createing a pandas DataFrame from the input data
                patient_data = pd.DataFrame([patient_details], columns=columns)

                # Scaling the data using the loaded scaler
                patient_data_scaled = scaler.transform(patient_data[scaler.feature_names_in_])

                # Making a prediction using the loaded model
                prediction = model.predict(patient_data_scaled)

                # Displaying the prediction
                st.write("Prediction:",
                         "Likely to have heart disease" if prediction[0] == 1 else "Not likely to have heart disease")
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.error("The app cannot proceed because the model or scaler failed to load.")
    st.write("Please check the model and scaler files and try again.")

# Adding a section for error handling
st.write("Error Handling:")
st.write("Please ensure that all fields are filled in correctly. If an error occurs, please try again.")

# Add a section for documentation
st.write("Documentation:")
st.write("This app uses a trained Random Forest Classifier model to predict the likelihood of heart disease based on patient details.")
st.write("The model was trained on a dataset of patient details and heart disease outcomes.")
st.write("The app uses Streamlit to create a user-friendly interface for inputting patient details and displaying the prediction.")
