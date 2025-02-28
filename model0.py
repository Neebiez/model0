import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import streamlit as st

data = pd.read_csv('netflix_titles0.csv')

data = data.dropna(axis=1, how='all')

label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

X = data.drop(columns=['type'])
y = data['type']

numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

st.title('Netflix Movie Type Prediction')

title = st.text_input('Title')
director = st.text_input('Director')
country = st.text_input('Country')
date_added = st.text_input('Date Added')
release_year = st.number_input('Release Year', min_value=1900, max_value=2023, step=1)
rating = st.text_input('Rating')
duration = st.text_input('Duration')
listed_in = st.text_input('Listed In')
description = st.text_area('Description')

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

user_data[numerical_cols] = imputer_num.transform(user_data[numerical_cols])
user_data[categorical_cols] = imputer_cat.transform(user_data[categorical_cols])

user_data = pd.get_dummies(user_data)

user_data = user_data.reindex(columns=X.columns, fill_value=0)

user_data = scaler.transform(user_data)

if st.button('Predict'):
    prediction = model.predict(user_data)
    prediction_label = label_encoder.inverse_transform(prediction)
    st.write(f'The predicted type is: {prediction_label[0]}')
