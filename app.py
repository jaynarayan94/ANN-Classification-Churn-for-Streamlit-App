import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Function to load files (model, encoders, scaler)
@st.cache_resource
def load_model(file_path):
    return tf.keras.models.load_model(file_path)

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load model and pre-processing objects
model = load_model('model.h5')
label_encoder_gender = load_pickle('label_encoder_gender.pkl')
onehot_encoder_geo = load_pickle('onehot_encoder_geo.pkl')
scaler = load_pickle('scaler.pkl')

# Streamlit App
st.title('Customer Churn Prediction')

# User input section with better input validations
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, value=30)  # Default value added for convenience
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format='%f')
tenure = st.slider('Tenure', 0, 10, value=5)  # Default value added
num_of_products = st.slider('Number of Products', 1, 4, value=1)
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

# Prepare input data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
    'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded geography with other input data
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict churn
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]

# Display prediction results
st.write(f'Churn Probability: {churn_probability:.2f}')
if churn_probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
