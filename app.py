import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

st.title("Salary Prediction App")  

# Load Dataset  
df = pd.read_csv("Salary_Prediction_Dataset.csv")  

# Data Cleaning  
df.dropna(inplace=True)  
df.drop_duplicates(inplace=True)  
df = df[df['Education_Level'].isin([1, 2, 3])]  
df = df[df['Company_Size'].isin([1, 2, 3])]  

# Feature & Target  
X = df[['Experience', 'Age', 'Education_Level', 'Company_Size']]  
y = df['Salary_LPA']  

# Model Training  
model = LinearRegression()  
model.fit(X, y)  

st.header("Enter Details to Predict Salary")  
experience = st.number_input("Experience (years)", min_value=0.0, step=0.5)  
age = st.number_input("Age", min_value=18, max_value=65, step=1)  
education_level = st.selectbox("Education Level", ["Bachelors (1)", "Masters (2)", "PhD (3)"])  
company_size = st.selectbox("Company Size", ["Small (1)", "Medium (2)", "Large (3)"])  

# Map selections to integers  
education_map = {"Bachelors (1)": 1, "Masters (2)": 2, "PhD (3)": 3}  
company_map = {"Small (1)": 1, "Medium (2)": 2, "Large (3)": 3}  

# Prediction  
if st.button("Predict Salary"):  
    input_features = np.array([[experience, age, education_map[education_level], company_map[company_size]]])  
    predicted_salary = model.predict(input_features)[0]  
    st.success(f"Predicted Salary: {predicted_salary:.2f} LPA")  

st.header("Model Performance")  

# Scatter Plot  
predictions = model.predict(X)  
fig, ax = plt.subplots()  
ax.scatter(y, predictions, color='green', label='Predictions')  
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='dashed', label='Ideal Line')  
ax.set_xlabel("Actual Salary")  
ax.set_ylabel("Predicted Salary")  
ax.set_title("Actual vs Predicted Salary")  
ax.grid(True)  
ax.legend()  
st.pyplot(fig)  

# Error Metrics  
mae = mean_absolute_error(y, predictions)  
mse = mean_squared_error(y, predictions)  
rmse = np.sqrt(mse)  
r2 = r2_score(y, predictions)  

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f} LPA")  
st.write(f"**Mean Squared Error (MSE):** {mse:.2f} LPA²")  
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f} LPA")  
st.write(f"**R² Score:** {r2:.2f}")  
