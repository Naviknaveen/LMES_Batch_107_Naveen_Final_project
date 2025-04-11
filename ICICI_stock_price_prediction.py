"""Prediction of ICICI bank stocks closing price"""

"""Step 1: Importing Required Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
file_path = "icici_dataset.csv"
df = pd.read_csv(file_path)
df.info()

"""Step 2: Cleaning and Preprocessing Data"""

# Trim column names and remove spaces
df.columns = df.columns.str.strip()

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Clean numeric columns and convert to float
num_cols = ["OPEN", "HIGH", "LOW", "PREV. CLOSE", "ltp", "close", "vwap", "VOLUME", "VALUE"]

for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

print(df.info())

print(df.isna().values.any())

"""Step 3: Data Visualization"""
# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Date", "OPEN", "HIGH", "LOW", "PREV. CLOSE", "close", "VOLUME", "VALUE"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Set plot style
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

"""Step 4: Time Series Visualization"""
# Create a figure and a 3x1 grid of subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# First subplot for OPEN and CLOSE prices
axs[0].plot(df['Date'], df["OPEN"], label='OPEN')
axs[0].plot(df['Date'], df["close"], label='CLOSE')
axs[0].set_title('Stock Opening and Closing Prices')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price in Rupees')
axs[0].legend()

# Second subplot for HIGH and LOW prices
axs[1].plot(df['Date'], df["HIGH"], label='HIGH')
axs[1].plot(df['Date'], df["LOW"], label='LOW')
axs[1].set_title('Stock High and Low Prices')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price in Rupees')
axs[1].legend()

# Third subplot for VOLUME and VALUE
axs[2].plot(df['Date'], df["VOLUME"], label='VOLUME')
axs[2].plot(df['Date'], df["VALUE"], label='VALUE')
axs[2].set_title('Stock Volume and Value')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Value')
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plots
plt.show()

"""Step 5: Unsupervised Learning: PCA"""
# Select features for PCA
features = ["HIGH", "LOW", "VOLUME", "VALUE"]
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Visualizing PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2")
plt.title("PCA Visualization of Stock Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

"""Step 6: Supervised Learning: Random Forest Regression"""

# Select features and target for regression
features = ["Year", "Month", "Day", "PCA1", "PCA2"]
target = "close"

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Train set graph (Adjusted for proper visualization)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train)), y_train, edgecolor='w', label='Actual Price')
plt.plot(range(len(y_train)), model.predict(X_train), color='r', label='Predicted Price')
plt.title('Linear Regression | Price vs Time')
plt.xlabel('Time Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


"""Step 7: Feature Selection Without PCA & Preprocessing"""

# Select features and target
features = ["Year", "Month", "Day", "HIGH", "LOW", "VOLUME"]
target = "close"

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""Step 8: Model Training"""

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""Step 9: Making Predictions & Evaluating Performance"""

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

print("When applying PCA, the model achieved an R² score of 0.85. However, after removing PCA and using the original features, the R² score improved to 0.98.")

# Train set graph (Adjusted for proper visualization)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train)), y_train, edgecolor='w', label='Actual Price')
plt.plot(range(len(y_train)), model.predict(X_train), color='r', label='Predicted Price')
plt.title('Linear Regression | Price vs Time')
plt.xlabel('Time Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

"""Step 10: Saving the Model"""

import joblib

# Save the trained model
joblib.dump(model, "stock_price_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("Next step is to run the app.py file")