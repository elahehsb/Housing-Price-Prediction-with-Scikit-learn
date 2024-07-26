# ---------------------------------
# 1. Import Libraries
# ---------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------
# Load and Explore the Dataset
# ---------------------------------

# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

# Display the first few rows of the dataset
print(data.head())

# Display dataset statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# ---------------------------------
# 3. Data Visualization
# ---------------------------------

# Correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(data['MEDV'], bins=30, kde=True)
plt.xlabel('MEDV')
plt.title('Distribution of MEDV')
plt.show()

# ---------------------------------
# 4. Feature Selection and Preprocessing
# ---------------------------------

# Define features and target variable
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# 5. Build and Train the Model
# ---------------------------------

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)

# ---------------------------------
# 6. Evaluate the Model
# ---------------------------------

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train MSE: {train_mse:.2f}, Train R2: {train_r2:.2f}')
print(f'Test MSE: {test_mse:.2f}, Test R2: {test_r2:.2f}')

# ---------------------------------
# 7. Visualization of Predictions
# ---------------------------------

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.show()



