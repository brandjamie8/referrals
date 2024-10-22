import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.dummy import DummyRegressor
from datetime import timedelta
# Title
st.title("Time Series Prediction for Referrals")
# Upload the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file with 'date' and 'referrals' columns)", type="csv")
if uploaded_file is not None:
   # Load the dataset
   data = pd.read_csv(uploaded_file)
   data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
   data.set_index('date', inplace=True)
   st.write("### Data preview:")
   st.write(data.head())
   # Plot the time series
   st.write("### Time Series Plot:")
   plt.figure(figsize=(10, 6))
   plt.plot(data.index, data['referrals'], label="Referrals")
   plt.title("Daily Referrals")
   plt.xlabel("Date")
   plt.ylabel("Referrals")
   plt.grid()
   st.pyplot(plt)
   # Train/Test Split
   split_ratio = 0.8
   train_size = int(len(data) * split_ratio)
   train, test = data.iloc[:train_size], data.iloc[train_size:]
   st.write("### Train/Test Split:")
   st.write(f"Train size: {train.shape[0]} days, Test size: {test.shape[0]} days")
   # Define models
   models = {
       "Linear Regression": LinearRegression(),
       "ARIMA": ARIMA(train['referrals'], order=(5, 1, 0)),
       "Exponential Smoothing": ExponentialSmoothing(train['referrals'], trend="add", seasonal="add", seasonal_periods=7),
       "Baseline (Mean)": DummyRegressor(strategy="mean")
   }
   predictions = {}
   errors = {}
   # Train and predict with each model
   for name, model in models.items():
       if name == "ARIMA":
           model_fit = model.fit()
           y_pred = model_fit.forecast(steps=len(test))
       elif name == "Exponential Smoothing":
           model_fit = model.fit()
           y_pred = model_fit.forecast(steps=len(test))
       else:
           model.fit(np.arange(len(train)).reshape(-1, 1), train['referrals'])
           y_pred = model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))
       predictions[name] = y_pred
       errors[name] = {
           "MAE": mean_absolute_error(test['referrals'], y_pred),
           "MSE": mean_squared_error(test['referrals'], y_pred)
       }
   # Choose the best model based on MAE
   best_model = min(errors, key=lambda x: errors[x]['MAE'])
   st.write(f"### Best Model: {best_model}")
   st.write(f"MAE: {errors[best_model]['MAE']}")
   st.write(f"MSE: {errors[best_model]['MSE']}")
   # Plot predictions for each model
   st.write("### Model Predictions:")
   plt.figure(figsize=(10, 6))
   plt.plot(test.index, test['referrals'], label="Actual", color="black", linestyle="--")
   for name, y_pred in predictions.items():
       plt.plot(test.index, y_pred, label=f"{name} Prediction")
   plt.title("Model Predictions vs Actual Referrals")
   plt.xlabel("Date")
   plt.ylabel("Referrals")
   plt.legend()
   plt.grid()
   st.pyplot(plt)
   # Predict the next year (365 days) with the best model
   if best_model == "ARIMA":
       final_model = models[best_model].fit()
       future_pred = final_model.forecast(steps=365)
   elif best_model == "Exponential Smoothing":
       final_model = models[best_model].fit()
       future_pred = final_model.forecast(steps=365)
   else:
       final_model = models[best_model]
       future_pred = final_model.predict(np.arange(len(data), len(data) + 365).reshape(-1, 1))
   future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=365)
   future_df = pd.DataFrame(future_pred, index=future_dates, columns=["Predicted Referrals"])
   st.write("### Next Year's Prediction:")
   st.write(future_df.head())
   # Plot future prediction
   st.write("### Next Year Referrals Prediction Plot:")
   plt.figure(figsize=(10, 6))
   plt.plot(future_dates, future_pred, label="Predicted Referrals")
   plt.title("Next Year's Predicted Referrals")
   plt.xlabel("Date")
   plt.ylabel("Referrals")
   plt.grid()
   st.pyplot(plt)
   # Display evaluation summary
   st.write("### Evaluation Summary:")
   st.write(f"The best performing model is **{best_model}** with:")
   st.write(f"- **Mean Absolute Error (MAE)**: {errors[best_model]['MAE']}")
   st.write(f"- **Mean Squared Error (MSE)**: {errors[best_model]['MSE']}")
   st.write("This model is used for the next year’s prediction.")
