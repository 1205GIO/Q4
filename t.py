import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
try:
    model = joblib.load("heart_disease_model.joblib")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Define the columns for the patient details
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Create a Streamlit app
st.title("Heart Disease Prediction App")
st.write("Enter the patient details to predict the likelihood of heart disease:")

# Create a form to input patient details
form = st.form("patient_details")
patient_details = {}
for column in columns:
    if column == 'sex':
        patient_details[column] = form.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    elif column == 'cp':
        patient_details[column] = form.selectbox('cp (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal Pain, 4 = Asymptomatic)', [1, 2, 3, 4])
    elif column == 'fbs':
        patient_details[column] = form.selectbox('fbs (> 120 mg/dl) (0 = False, 1 = True)', [0, 1])
    elif column == 'exang':
        patient_details[column] = form.selectbox('exang (0 = No, 1 = Yes)', [0, 1])
    else:
        patient_details[column] = form.number_input(column.capitalize())

submit = form.form_submit_button("Submit")

if submit:
    st.write("Patient Details:")
    for column, value in patient_details.items():
        st.write(f"{column.capitalize()}: {value}")

    # Process the input data
    try:
        # Create a pandas DataFrame from the input data
        patient_data = pd.DataFrame([patient_details], columns=columns)

        # Scale the data
        scaler = StandardScaler()
        patient_data_scaled = scaler.fit_transform(patient_data)

        # Make a prediction using the loaded model
        prediction = model.predict(patient_data_scaled)

        # Display the prediction
        st.write("Prediction:",
                 "Likely to have heart disease" if prediction[0] == 1 else "Not likely to have heart disease")
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a section for error handling
st.write("Error Handling:")
st.write("Please ensure that all fields are filled in correctly. If an error occurs, please try again.")

# Add a section for documentation
st.write("Documentation:")
st.write("This app uses a trained Random Forest Classifier model to predict the likelihood of heart disease based on patient details.")
st.write("The model was trained on a dataset of patient details and heart disease outcomes.")
st.write("The app uses Streamlit to create a user-friendly interface for inputting patient details and displaying the prediction.")
