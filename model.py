import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (update the path if needed)
df = pd.read_csv("housing.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Select features and target variable
X = df[['median_income']]  # Using 'median_income' as predictor
y = df['median_house_value']  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Model Coefficients: {model.coef_}")  # Slope
print(f"Model Intercept: {model.intercept_}")  # Intercept



# Plot results
plt.scatter(X_test, y_test, color='blue', label="Actual Prices")
plt.plot(X_test, y_pred, color='red', label="Prediction Line")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.legend()
plt.show()

joblib.dump(model, "linear_regression_model.joblib")
print("Model saved successfully!")
