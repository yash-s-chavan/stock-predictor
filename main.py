import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import json
import os
TICKER = "NVDA"
LOOKBACK_DAYS = 60
TRAINING_YEARS = 5


def fetch_and_prep_data(ticker):
    print(f"Fetching {TRAINING_YEARS} years of data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{TRAINING_YEARS}y")

    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return data, scaled_data, scaler


def build_and_train_model(scaled_data):
    print("Prepping training sequences...")
    x_train, y_train = [], []

    for i in range(LOOKBACK_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i - LOOKBACK_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("Building LSTM Model Architecture...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training Model (This may take a minute...)")
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    model.save('latest_model.keras')
    print("Model saved to latest_model.keras")

    return model


def make_prediction(model, scaler, data, scaled_data):
    last_60_days = scaled_data[-LOOKBACK_DAYS:]
    X_test = np.array([last_60_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    current_price = data[-1][0]
    prediction = predicted_price[0][0]

    return float(current_price), float(prediction)


def main():
    data, scaled_data, scaler = fetch_and_prep_data(TICKER)
    model = build_and_train_model(scaled_data)
    current_price, predicted_price = make_prediction(model, scaler, data, scaled_data)

    threshold = 0.01
    if predicted_price > current_price * (1 + threshold):
        signal = "BUY"
    elif predicted_price < current_price * (1 - threshold):
        signal = "SELL"
    else:
        signal = "HOLD"

    output = {
        "ticker": TICKER,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "signal": signal
    }

    print("\n" + "=" * 40)
    print("FINAL API PAYLOAD:")
    print(json.dumps(output, indent=2))
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()