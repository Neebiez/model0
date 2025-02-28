import pandas as pd
import streamlit as st
import pickle

# Load the model and preprocessors from the pickle file
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    imputer_num = data['imputer_num']
    imputer_cat = data['imputer_cat']
    label_encoder = data['label_encoder']
    columns = data['columns']
    numerical_cols = data['numerical_cols']
    categorical_cols = data['categorical_cols']

# Streamlit part for user input
st.title('Netflix Movie Type Prediction')

# Input fields for user
title = st.text_input('Title')
director = st.text_input('Director')
country = st.text_input('Country')
date_added = st.text_input('Date Added')
release_year = st.number_input('Release Year', min_value=1900, max_value=2023, step=1)
rating = st.text_input('Rating')
duration = st.text_input('Duration')
listed_in = st.text_input('Listed In')
description = st.text_area('Description')

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'title': [title],
    'director': [director],
    'country': [country],
    'date_added': [date_added],
    'release_year': [release_year],
    'rating': [rating],
    'duration': [duration],
    'listed_in': [listed_in],
    'description': [description]
})

# Handle missing values for user input
user_data[numerical_cols] = imputer_num.transform(user_data[numerical_cols])
user_data[categorical_cols] = imputer_cat.transform(user_data[categorical_cols])

# Handle categorical features if any (one-hot encoding)
user_data = pd.get_dummies(user_data)

# Align user_data with training data
user_data = user_data.reindex(columns=columns, fill_value=0)

# Standardize the user input
user_data = scaler.transform(user_data)

# Predict using the Decision Tree model
if st.button('Predict'):
    prediction = model.predict(user_data)
    prediction_label = label_encoder.inverse_transform(prediction)
    st.write(f'The predicted type is: {prediction_label[0]}')
