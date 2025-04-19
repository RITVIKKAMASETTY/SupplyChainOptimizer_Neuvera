# Supply Chain Optimizer

A machine learning-based tool for optimizing supply chain operations, forecasting demand, and identifying bottlenecks.

## Overview

This project uses deep learning (LSTM) to analyze and forecast supply chain metrics such as resource usage and utility allocation. It helps supply chain managers make data-driven decisions to optimize operations and prevent bottlenecks.

## Features

- **Data Exploration**: Visualize and analyze supply chain data
- **LSTM Model Training**: Train deep learning models to understand patterns in supply chain data
- **Demand Forecasting**: Predict future resource needs and utility allocation
- **Supply Chain Insights**: Identify bottlenecks and optimization opportunities

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

Launch the interactive dashboard with:

```bash
streamlit run DemandSupply/supply_chain_app.py
```

### Using the DemandForecaster Module

You can also use the demand forecasting module programmatically:

```python
from DemandSupply.demand_forecast import DemandForecaster

# Initialize the forecaster
forecaster = DemandForecaster('DemandSupply/data/ecommerce_supply_chain_data.csv')

# Load and preprocess data
data = forecaster.load_data()
processed_data = forecaster.preprocess_data()

# Define features and target columns
features = [
    'resource_usage', 'utility_allocation', 'shelf_life', 'hour', 'day', 'month',
    'day_of_week', 'days_to_deadline', 'department_encoded', 'priority_encoded',
    'departure_loc_encoded', 'arrival_loc_encoded', 'return_status_encoded',
    'festive_season_encoded', 'status_encoded'
]

target_cols = ['resource_usage', 'utility_allocation']

# Create sequences and train the model
X, y = forecaster.create_sequences(features, target_cols, lookback=3)
history = forecaster.train(X_train, y_train)

# Generate a forecast
forecasts = forecaster.forecast(sequence, target_cols, n_steps=7)
```

## Dashboard Tabs

The Streamlit application includes the following tabs:

1. **Data Exploration**: Visualize the supply chain data with interactive charts and tables
2. **Model Training**: Configure and train the LSTM model with custom parameters
3. **Forecasting**: Generate and visualize forecasts for different departments
4. **Insights**: View key performance indicators and recommendations for optimization

## Data Requirements

The application expects a CSV file with the following columns:

- `department`: Department category (Logistics, Inventory, etc.)
- `time`: Timestamp of the record
- `resource_usage`: Resource utilization (numeric)
- `utility_allocation`: Utility allocation (numeric)
- `priority`: Priority level (numeric)
- `departure_loc`: Departure location
- `arrival_loc`: Arrival location
- `deadline`: Deadline timestamp
- `return_status`: Return status (Yes/No)
- `festive_season`: Festive season flag (Yes/No)
- `status`: Current status (In Transit, Delivered, etc.)
- `bottleneck_flag`: Bottleneck indicator (0/1)

## Model Architecture

The LSTM neural network architecture used for forecasting includes:

- Input layer with lookback window
- LSTM layer with 64 units and ReLU activation
- Dropout layer (20%)
- LSTM layer with 32 units and ReLU activation
- Dropout layer (20%)
- Dense output layer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.



