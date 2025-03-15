from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load California Housing Dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Features
y = data.target  # Target (House Prices)

# Check for missing values
print("Missing values in each feature:")
print(X.isnull().sum())

# Fill missing values with mean (if any)
X.fillna(X.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
git add train_california_housing.py
# Predict using Linear Regression Model
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - Mean Squared Error: {mse_lr:.4f}")
print(f"Linear Regression - R² Score: {r2_lr:.4f}")

# Plot Actual vs Predicted House Prices (Linear Regression)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.legend()
plt.show()

# Train Decision Tree Regressor Model
dt_model = DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=16, random_state=42)
dt_model.fit(X_train, y_train)

# Predict using Decision Tree Regressor
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree - Mean Squared Error: {mse_dt:.4f}")
print(f"Decision Tree - R² Score: {r2_dt:.4f}")

# Plot Actual vs Predicted House Prices (Decision Tree)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_dt, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
plt.title('Actual vs Predicted House Prices (Decision Tree)')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.legend()
plt.show()

# Plot Decision Tree
plt.figure(figsize=(16, 15))
plot_tree(dt_model, filled=True, feature_names=X.columns, fontsize=10)
plt.show()

# Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
