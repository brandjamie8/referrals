import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
# Title
st.title("Time Series Prediction for Referrals")
# Frequency selection
frequency = st.selectbox("Select the frequency of your data:", ["Daily", "Weekly", "Monthly"])
# Upload the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file with 'date' and 'referrals' columns)", type="csv")
if uploaded_file is not None:
   # Load the dataset
   data = pd.read_csv(uploaded_file)
   data['date'] = pd.to_datetime(data['date'])
   data.set_index('date', inplace=True)
   # Resample based on the frequency
   if frequency == "Daily":
       data = data.resample('D').sum()
   elif frequency == "Weekly":
       data = data.resample('W').sum()
   elif frequency == "Monthly":
       data = data.resample('M').sum()
   st.write("### Data preview:")
   st.write(data.head())
   # Plot the time series
   st.write("### Time Series Plot:")
   plt.figure(figsize=(10, 6))
   plt.plot(data.index, data['referrals'], label="Referrals")
   plt.title(f"{frequency} Referrals")
   plt.xlabel("Date")
   plt.ylabel("Referrals")
   plt.grid()
   st.pyplot(plt)
   # Train/Test Split
   split_ratio = 0.8
   train_size = int(len(data) * split_ratio)
   train, test = data.iloc[:train_size], data.iloc[train_size:]
   st.write("### Train/Test Split:")
   st.write(f"Train size: {train.shape[0]} periods, Test size: {test.shape[0]} periods")
   # Model 1: Prophet
   prophet_data = data.reset_index().rename(columns={'date': 'ds', 'referrals': 'y'})
   prophet_train = prophet_data.iloc[:train_size]
   prophet_test = prophet_data.iloc[train_size:]
   prophet_model = Prophet()
   prophet_model.fit(prophet_train)
   future = prophet_model.make_future_dataframe(periods=len(test), freq=frequency[0])
   prophet_forecast = prophet_model.predict(future)
   y_pred_prophet = prophet_forecast['yhat'].iloc[-len(test):].values
   # Model 2: LSTM
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_train = scaler.fit_transform(train)
   X_train, y_train = [], []
   for i in range(60, len(scaled_train)):
       X_train.append(scaled_train[i - 60:i, 0])
       y_train.append(scaled_train[i, 0])
   X_train, y_train = np.array(X_train), np.array(y_train)
   # Reshape input to be [samples, time steps, features]
   X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
   lstm_model = Sequential()
   lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
   lstm_model.add(LSTM(units=50))
   lstm_model.add(Dense(1))
   lstm_model.compile(optimizer='adam', loss='mean_squared_error')
   lstm_model.fit(X_train, y_train, epochs=5, batch_size=32)
   total_data = pd.concat((train, test), axis=0)
   inputs = total_data[len(total_data) - len(test) - 60:].values
   inputs = inputs.reshape(-1, 1)
   inputs = scaler.transform(inputs)
   X_test = []
   for i in range(60, len(inputs)):
       X_test.append(inputs[i - 60:i, 0])
   X_test = np.array(X_test)
   X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
   y_pred_lstm = lstm_model.predict(X_test)
   y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
   # Model 3: Random Forest
   X = np.arange(len(train)).reshape(-1, 1)
   y = train['referrals'].values
   rf_model = RandomForestRegressor(n_estimators=100)
   rf_model.fit(X, y)
   X_test_rf = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
   y_pred_rf = rf_model.predict(X_test_rf)
   # Baseline model: Mean prediction
   y_pred_baseline = np.full_like(test['referrals'], train['referrals'].mean())
   # Collect all predictions
   predictions = {
       "Prophet": y_pred_prophet,
       "LSTM": y_pred_lstm.flatten(),
       "Random Forest": y_pred_rf,
       "Baseline (Mean)": y_pred_baseline
   }
   errors = {}
   for name, y_pred in predictions.items():
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
   # Predict the next year (based on selected frequency)
   if best_model == "Prophet":
       future_pred = prophet_model.predict(prophet_model.make_future_dataframe(periods=52, freq=frequency[0]))['yhat'][-52:].values
   elif best_model == "LSTM":
       future_pred = lstm_model.predict(X_test)
       future_pred = scaler.inverse_transform(future_pred)[-52:].flatten()
   elif best_model == "Random Forest":
       future_pred = rf_model.predict(np.arange(len(data), len(data) + 52).reshape(-1, 1))
   future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(1, 'D'), periods=52, freq=frequency[0])
   future_df = pd.DataFrame(future_pred, index=future_dates, columns=["Predicted Referrals"])
   st.write("### Next Year Prediction:")
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
