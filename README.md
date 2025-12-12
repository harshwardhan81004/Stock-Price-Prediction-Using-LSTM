# Stock-Price-Prediction-Using-LSTM

This project provides a modular, reusable pipeline for **predicting stock prices** using a **Stacked LSTM model**.  
The flow is split cleanly across three Python modules:

- `preprocessing.py`
- `model_builder.py`
- `forecasting.py`

and a main Jupyter Notebook (`main.ipynb`) that ties together the entire workflow and generates all plots.

This structure allows the same pipeline to work on **any stock dataset** with minimal adjustments.

# Full Modularity of the Project

One of the core design principles of this project is **complete modularity**.

### All core logic lives in `.py` modules  
You can modify functions inside:

- `preprocessing.py`  
- `model_builder.py`  
- `forecasting.py`  

to change **how the model behaves**, without touching the notebook.

### The notebook only controls:  
- Input parameters (e.g., `time_step`, epochs, future_days)  
- Which ticker/dataset to load  
- Which functions to call  

### Users can change:  
- LSTM architecture  
- Number of layers  
- Number of neurons  
- Sequence length (time_step)  
- Forecasting horizon  
- Train-test ratio  
- Plotting styles  
- Preprocessing steps  

All by editing the `.py` modules — no duplication of code inside the notebook.

### `main.ipynb` becomes a clean “controller”  
It simply:

1. Loads data  
2. Calls preprocessing functions  
3. Calls model builder functions  
4. Calls forecasting functions  
5. Generates plots  

This keeps the project clean, professional, scalable, and easily adaptable.

## Project Structure
```
StockPricePrediction/
│
├── preprocessing.py
├── model_builder.py
├── forecasting.py
├── main.ipynb
└── README.md
```

# 1. preprocessing.py

Handles **data cleaning, scaling, splitting, and sequence creation**.

### **Key Functions**

### **`plot_initial_series(series)`**
Plots the **raw dataset before any model or scaling is applied**.  
Helps visualize overall trends and ensure data is correct.

### **`scale_series(series)`**
- Scales data using MinMaxScaler (range 0–1)
- Returns both scaled values and the scaler object  
Used before training the LSTM model.


### **`split_train_test(scaled_data, train_ratio=0.65)`**
Splits the scaled dataset into:

- **Training set (65%)**
- **Testing set (35%)**

### **`create_sequences(dataset, time_step)`**
Converts the 1-D time series into supervised learning input/output pairs:

- Inputs: sequences of length `time_step`
- Output: next value in the series

This is required for LSTM training.


### **`clean_stock_dataframe(df)` (optional)**
Useful for messy CSV files downloaded from Yahoo Finance.  
Cleans the dataset by:

- Removing unnecessary rows  
- Converting columns to numeric  
- Removing NaNs  
- Resetting index  


# 2. model_builder.py

Defines and trains the **Stacked LSTM model**.

### **Key Functions**


### **`build_lstm_model(time_step, n_units, n_layers)`**
Builds an LSTM architecture with:

- Configurable number of layers  
- Adjustable number of units  
- Automatically sets correct input shape  

Outputs a model ready for training.


### **`train_model(model, X_train, y_train, X_val, y_val)`**
Trains the LSTM with validation support.

Parameters include:

- Number of epochs  
- Batch size  
- Verbosity  

Returns the training history.


# 3. forecasting.py

Responsible for all **predictions, future forecasting, and visualizations**.

### **Key Functions**

### **`predict_and_inverse(model, X_train, X_test, scaler)`**
- Generates predictions for both training and test sets  
- Inverse-transforms them back to actual price values  
- Returns both series

### **`plot_train_predictions(train_inv)`**
Plots **only the training predictions**.

### **`plot_test_predictions(test_inv)`**
Plots **only the test predictions**.

### **`forecast_next_days(model, test_data, scaler, time_step, n_days)`**
Performs auto-regressive forecasting:

1. Takes last `time_step` values  
2. Predicts the next value  
3. Appends the prediction  
4. Repeats for `n_days`  

Returns inverse-transformed future forecast values.

### **`plot_future_window(test_data, future_pred, scaler, time_step)`**
Creates the **zoomed future forecast plot**, showing:

- Last real `time_step` values in **blue**
- Next predicted values in **green**
- Continuous curve without gaps  

This resembles real financial forecast charts.

### **`plot_results(...)`**
Generates the **final combined visualization**:

- Full original dataset (light gray)
- Train predictions (blue)
- Test predictions (red)
- Future forecasting (green)
- Clear shaded regions marking phases

This is the complete summary of the model’s performance.

# 4. main.ipynb — Full Workflow

The notebook executes the complete end-to-end pipeline:


