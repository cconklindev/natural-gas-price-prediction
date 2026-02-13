"""
Natural Gas Price Prediction Application
Author: Christine Conklin
Purpose: Predict next-day natural gas closing price using Random Forest Regression
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Download Natural Gas Data
# -----------------------------
def load_data():
    data = yf.download("NG=F", start="2010-01-01", end="2024-01-01")
    data.reset_index(inplace=True)
    return data

# -----------------------------
# 2. Feature Engineering
# -----------------------------
def create_features(df):
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)
    return df

# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(df):
    features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Volatility', 'Return']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions

# -----------------------------
# 4. Evaluate Model
# -----------------------------
def evaluate(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# -----------------------------
# 5. Plot Results
# -----------------------------
def plot_results(y_test, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title("Natural Gas Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# -----------------------------
# 6. Main
# -----------------------------
def main():
    print("Downloading Natural Gas Data...")
    df = load_data()

    print("Creating Features...")
    df = create_features(df)

    print("Training Model...")
    model, X_test, y_test, predictions = train_model(df)

    evaluate(y_test, predictions)
    plot_results(y_test, predictions)

if __name__ == "__main__":
    main()
