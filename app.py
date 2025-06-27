import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
df = pd.read_csv('DataScience_Salary_vs_Experience.csv')

# Train model
model = LinearRegression()
model.fit(df[['Years_of_Experience']], df['Salary_LPA'])


# Streamlit App
st.title("Data Science Salary Prediction")

# User input
experience = st.number_input("Enter your years of experience:", min_value=0.0, step=0.1)

if st.button("Predict Salary"):
    input_df = pd.DataFrame({'Years_of_Experience': [experience]})
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary: {prediction[0]:.2f} LPA")
