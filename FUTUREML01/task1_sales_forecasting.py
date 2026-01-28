# ======================================
# TASK 1: SALES FORECASTING - SUPERSTORE
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  Load Dataset
data = pd.read_csv("Sample - Superstore.csv", encoding="latin1")


print("Dataset Loaded Successfully")
print(data.head())

#  Select Required Columns
data = data[['Order Date', 'Sales']]

# Convert Order Date to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'])

#  Aggregate Daily Sales
daily_sales = data.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.sort_values('Order Date')

print("\nDaily Sales Preview:")
print(daily_sales.head())

#  Feature Engineering
daily_sales['Day'] = daily_sales['Order Date'].dt.day
daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['Year'] = daily_sales['Order Date'].dt.year
daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek

X = daily_sales[['Day', 'Month', 'Year', 'DayOfWeek']]
y = daily_sales['Sales']

#  Train-Test Split (Time Series Safe)
split_index = int(len(daily_sales) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

#  Train Model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predictions
y_pred = model.predict(X_test)

#  Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot: Actual vs Predicted Sales
plt.figure(figsize=(10, 5))
plt.plot(daily_sales['Order Date'].iloc[split_index:], y_test.values, label="Actual Sales")
plt.plot(daily_sales['Order Date'].iloc[split_index:], y_pred, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

#  Forecast Next 30 Days
future_dates = pd.date_range(
    start=daily_sales['Order Date'].max(),
    periods=31,
    freq='D'
)[1:]

future_features = pd.DataFrame({
    'Day': future_dates.day,
    'Month': future_dates.month,
    'Year': future_dates.year,
    'DayOfWeek': future_dates.dayofweek
})

future_sales = model.predict(future_features)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': future_sales
})

print("\nNext 30 Days Sales Forecast:")
print(forecast_df.head())

#  Plot: Forecasted Sales
plt.figure(figsize=(10, 5))
plt.plot(daily_sales['Order Date'], daily_sales['Sales'], label="Historical Sales")
plt.plot(forecast_df['Date'], forecast_df['Forecasted Sales'], label="Forecasted Sales")
plt.title("Sales Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
