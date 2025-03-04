# Stock Market Prediction using LSTM
## Table of Contents
- [Overview](#-overview)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features](#-features)
- [Dependencies](#-dependencies)
- [Model Overview](#-model-overview)
- [Examples](#-examples)


## Overview
This project implements a **Stock Market Prediction** model using **Long Short-Term Memory (LSTM)** networks. The dataset consists of historical stock prices, and the goal is to predict future stock prices based on past trends. The dataset used is from **Tata Global Beverages**.



## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the following dependencies:

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

## Usage

1. **Load the Dataset**:
   The dataset is available at:
   ```
   https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv
   ```
   Load it using Pandas:
   ```python
   import pandas as pd
   data = pd.read_csv('NSE-TATAGLOBAL.csv')
   data = data.iloc[::-1]  # Reverse the dataset order
   ```

2. **Data Preprocessing**:
   - Handle missing values (if any)
   - Normalize data using `MinMaxScaler`
   - Split data into training and testing sets

3. **Train the LSTM Model**:
   - Define an LSTM network with multiple layers.
   - Train the model using `Adam` optimizer and `mean_squared_error` loss function.

4. **Make Predictions**:
   - Use the trained model to predict future stock prices.
   - Rescale predicted values to match the original price range.

5. **Evaluate the Model**:
   - Use **Mean Squared Error (MSE)** for model evaluation.
   - Plot predictions alongside actual stock prices for visualization.

## Features

- **LSTM-Based Prediction**: Uses an advanced RNN model for time series forecasting.
- **Data Normalization**: Scales stock prices for effective model training.
- **Train-Test Split**: Implements an 80-20 split for evaluation.
- **Graphical Visualization**: Uses Matplotlib for plotting trends and predictions.

## Dependencies

- `numpy` - For numerical operations.
- `pandas` - For data manipulation.
- `matplotlib` - For plotting graphs.
- `scikit-learn` - For data preprocessing.
- `tensorflow` - For deep learning (LSTM model).

## Model Overview

The **LSTM (Long Short-Term Memory)** model is used for stock price prediction due to its ability to handle sequential data efficiently.

### Model Architecture:
1. **Input Layer**: Takes past stock prices as input.
2. **LSTM Layers**: Extract sequential patterns.
3. **Dense Layer**: Outputs the predicted stock price.

### Example Model Code:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
```

## Example Usage

### Predicting Future Stock Prices

```python
import numpy as np
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Rescale values
print(predictions)
```



