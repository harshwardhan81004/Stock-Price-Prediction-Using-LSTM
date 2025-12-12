import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def clean_stock_dataframe(df, price_column="Close"):

    # 1. Drop rows where the price_column is not numeric
    # (Catches rows like 'Ticker', 'Date', etc.)
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    df = df[df[price_column].apply(is_float)]

    # 2. Convert price column to float
    df[price_column] = df[price_column].astype(float)

    # 3. Drop NaN rows
    df = df.dropna()

    # 4. Reset index
    df = df.reset_index(drop=True)

    return df


def scale_series(series, scaler=None, feature_range=(0, 1)):
    if isinstance(series, pd.Series):
        arr = series.values.reshape(-1, 1)  # type: ignore
    else:
        arr = np.array(series).reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)

    return scaled, scaler

def split_train_test(scaled_data, train_ratio=0.65):
    n = len(scaled_data)
    training_size = int(n * train_ratio)
    train = scaled_data[:training_size]
    test = scaled_data[training_size:]
    return train, test

def create_sequences(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def plot_initial_series(series, title="Original Stock Price Data"):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(series, color='black', linewidth=1.5, label='Original Data')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
