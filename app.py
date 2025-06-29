import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and clean the dataset
df = pd.read_csv('Salary_Prediction_Dataset.csv')

# Basic cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['Education_Level'].isin([1, 2, 3])]
df = df[df['Company_Size'].isin([1, 2, 3])]

# Train the model
X = df[['Experience', 'Age', 'Education_Level', 'Company_Size']]
y = df['Salary_LPA']

model = LinearRegression()
model.fit(X, y)

# Streamlit App UI
st.title("Advanced Data Science Salary Prediction")

st.header("Enter Your Details:")

experience = st.number_input("Experience (years):", min_value=0.0, step=0.1)
age = st.number_input("Age:", min_value=18, max_value=65, step=1)
education_level = st.selectbox("Education Level:", ["Bachelors (1)", "Masters (2)", "PhD (3)"])
company_size = st.selectbox("Company Size:", ["Small (1)", "Medium (2)", "Large (3)"])

# Mapping text inputs to numeric
education_level_num = int(education_level.split('(')[-1][0])
company_size_num = int(company_size.split('(')[-1][0])

if st.button("Predict Salary"):
    input_data = [[experience, age, education_level_num, company_size_num]]
    predicted_salary = model.predict(input_data)
    st.success(f"Predicted Salary: {predicted_salary[0]:.2f} LPA")

# Visualization
st.header("Model Performance")

predictions = model.predict(X)

plt.scatter(y, predictions, color='green')
plt.xlabel("Actual Salary (LPA)")
plt.ylabel("Predicted Salary (LPA)")
plt.title("Actual vs Predicted Salary")

line = np.linspace(min(y), max(y), 100)
plt.plot(line, line, color='red', linestyle='dashed')

plt.grid(True)
st.pyplot()

# Error Metrics
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f} LPA")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f} LPA²")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f} LPA")
st.write(f"**R² Score:** {r2:.2f}")
