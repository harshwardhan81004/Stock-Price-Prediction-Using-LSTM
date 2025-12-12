from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

def build_lstm_model(time_step, n_units=50, n_layers=3, dropout=0.0):
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_units, input_shape=(time_step, 1)))
    else:
        model.add(LSTM(n_units, return_sequences=True, input_shape=(time_step, 1)))
        for _ in range(n_layers - 2):
            model.add(LSTM(n_units, return_sequences=True))
        model.add(LSTM(n_units))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64, verbose=1):
    if X_val is not None and y_val is not None:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return history
