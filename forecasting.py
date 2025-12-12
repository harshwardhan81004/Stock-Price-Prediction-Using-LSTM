import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ======================================================================
#  SEPARATE PLOT FUNCTIONS (ALWAYS TRIGGERED WHEN CALLED)
# ======================================================================

def plot_train_predictions(train_inv):
    plt.figure(figsize=(8,4))
    plt.title("Train Predictions Only")
    plt.plot(train_inv, label="Train Predict", color='blue')
    plt.legend()
    plt.show()


def plot_test_predictions(test_inv):
    plt.figure(figsize=(8,4))
    plt.title("Test Predictions Only")
    plt.plot(test_inv, label="Test Predict", color='red')
    plt.legend()
    plt.show()


def plot_future_predictions(future_inv):
    plt.figure(figsize=(8,4))
    plt.title("Future Forecast Only")
    plt.plot(future_inv, label="Future Forecast", color='green')
    plt.legend()
    plt.show()


# ======================================================================
#  PREDICT TRAIN + TEST AND PLOT SEPARATELY
# ======================================================================

def predict_and_inverse(model, X_train, X_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_inv = scaler.inverse_transform(train_predict)
    test_inv = scaler.inverse_transform(test_predict)

    # ALWAYS plot
    plot_train_predictions(train_inv)
    plot_test_predictions(test_inv)

    return train_inv, test_inv


# ======================================================================
#  LAST WINDOW HELPER
# ======================================================================

def _get_last_window(test_data, time_step):
    if len(test_data) >= time_step:
        return test_data[-time_step:].reshape(1, -1)
    else:
        pad = np.zeros((time_step - len(test_data), 1))
        full = np.vstack([pad, test_data])
        return full.reshape(1, -1)


# ======================================================================
#  FUTURE FORECASTING + SEPARATE PLOT
# ======================================================================

def forecast_next_days(model, test_data, scaler, time_step=100, n_days=30):
    x_input = _get_last_window(test_data, time_step)
    temp_input = list(x_input.reshape(-1))

    lst_output = []

    for _ in range(n_days):
        x_arr = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
        yhat = model.predict(x_arr, verbose=0)
        temp_input.append(float(yhat[0, 0])) # type: ignore
        lst_output.append(yhat[0].tolist())

    future_scaled = np.array(lst_output).reshape(-1, 1)
    future_inv = scaler.inverse_transform(future_scaled)

    # ALWAYS plot
    # plot_future_predictions(future_inv)

    return future_inv


def plot_future_window(test_data, future_inv, scaler, time_step=100):
    """
    Plot last <time_step> actual values + future predictions as one continuous curve.
    No gap between actual and predicted.
    """

    # Inverse transform last window
    last_window = scaler.inverse_transform(test_data[-time_step:])

    # Create combined series: last_window + predictions
    combined = last_window.flatten().tolist() + future_inv.flatten().tolist()

    # X-axis for combined plot
    x_axis = range(len(combined))

    plt.figure(figsize=(10, 5))
    plt.title("Future Forecast (Continuous Plot)")

    # Plot entire combined line
    plt.plot(x_axis, combined, color="blue", label="Actual + Future")

    # Highlight the predicted part in green
    plt.plot(
        range(time_step, time_step + len(future_inv)),
        future_inv,
        color="green",
        label="Predicted Future"
    )

    plt.legend()
    plt.show()




# ======================================================================
#  FINAL COMBINED PLOT
# ======================================================================

def plot_results(scaled_all, train_predict, test_predict, future_predict, scaler, look_back=100):
    """
    Clean, professional combined plot:
    - light gray baseline
    - clear train/test/future segments
    - shaded regions
    """

    baseline = scaler.inverse_transform(scaled_all)

    # Prepare placeholders for train/test lines
    train_plot = np.empty_like(scaled_all)
    train_plot[:] = np.nan
    train_plot[look_back : len(train_predict) + look_back] = train_predict

    test_plot = np.empty_like(scaled_all)
    test_plot[:] = np.nan
    test_start = len(train_predict) + (look_back * 2) + 1
    test_end = test_start + len(test_predict)
    if test_start < len(test_plot):
        test_plot[test_start:test_end] = test_predict

    # Setup figure
    plt.figure(figsize=(14, 6))
    plt.title("Model Performance Overview (Train    Test    Future)", fontsize=14)

    # --- Baseline (light gray background reference)
    plt.plot(baseline, label="Original Data", color="gray", linewidth=1, alpha=0.6)

    # --- Train predictions (blue)
    plt.plot(train_plot, label="Train Predict", color="blue", linewidth=2)

    # --- Test predictions (red)
    plt.plot(test_plot, label="Test Predict", color="red", linewidth=2)

    # --- Future Forecast (green curve)
    future_x = range(len(baseline), len(baseline) + len(future_predict))
    plt.plot(future_x, future_predict, label="Future Forecast", color="green", linewidth=2)

    # --- Shaded Regions
    # Train region
    plt.axvspan(0, len(train_predict) + look_back, color='blue', alpha=0.05)

    # Test region
    plt.axvspan(len(train_predict) + look_back,
                len(train_predict) + look_back + len(test_predict) + look_back,
                color='red', alpha=0.05)

    # Future region
    plt.axvspan(len(baseline), len(baseline) + len(future_predict),
                color='green', alpha=0.1)

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

