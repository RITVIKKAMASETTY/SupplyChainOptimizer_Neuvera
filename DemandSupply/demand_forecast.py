from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_components
from prophet.diagnostics import cross_validation, performance_metrics
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and prepare the data
print("Step 1: Loading and preparing data...")
try:
    df = pd.read_csv('supply_chain_data.csv')
    print("CSV file loaded successfully")
    print("Columns found:", df.columns.tolist())
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'supply_chain_data.csv' was not found. Please ensure it's in the same directory.")

# Step 2: Data Preprocessing
print("\nStep 2: Data Preprocessing...")

# 2.1 Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='median')
numeric_cols = ['resource_usage', 'utility_allocation', 'shelf_life']
for col in numeric_cols:
    if col in df.columns:
        df[col] = imputer.fit_transform(df[[col]])

categorical_cols = ['department', 'priority', 'departure_loc', 'arrival_loc',
                    'return_status', 'festive_season', 'status']
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# 2.2 Convert date columns with flexible parsing
print("Converting date columns...")
date_columns = ['time', 'deadline']
for col in date_columns:
    if col in df.columns:
        try:
            # Try multiple date formats
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True)
        except Exception as e:
            print(f"Error converting {col}: {e}")
            # Try inferring format if mixed fails
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

# 2.3 Feature engineering
print("Creating new features...")
if 'time' in df.columns and 'deadline' in df.columns:
    df['days_to_deadline'] = (df['deadline'] - df['time']).dt.days
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
else:
    print("Warning: Required time columns not found for feature engineering")

# 2.4 Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

# 2.5 Normalize numerical features
print("Normalizing numerical features...")
scaler = MinMaxScaler()
numeric_features = [col for col in numeric_cols if col in df.columns]
if numeric_features:
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Step 3: Exploratory Data Analysis
print("\nStep 3: Exploratory Data Analysis...")

plt.figure(figsize=(18, 12))

if 'time' in df.columns and 'resource_usage' in df.columns:
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], df['resource_usage'], 'b-')
    plt.title('Resource Usage Over Time')
    plt.xticks(rotation=45)

if 'time' in df.columns and 'utility_allocation' in df.columns:
    plt.subplot(2, 2, 2)
    plt.plot(df['time'], df['utility_allocation'], 'g-')
    plt.title('Utility Allocation Over Time')
    plt.xticks(rotation=45)

if 'department' in df.columns and 'resource_usage' in df.columns:
    plt.subplot(2, 2, 3)
    sns.boxplot(x='department', y='resource_usage', data=df)
    plt.title('Resource Usage by Department')
    plt.xticks(rotation=45)

if 'priority' in df.columns and 'utility_allocation' in df.columns:
    plt.subplot(2, 2, 4)
    sns.boxplot(x='priority', y='utility_allocation', data=df)
    plt.title('Utility Allocation by Priority')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('exploratory_analysis.png')
print("Exploratory analysis saved")

# Step 4: Prepare Prophet data
print("\nStep 4: Preparing Prophet data...")
if 'time' in df.columns and 'resource_usage' in df.columns:
    df_resource = df[['time', 'resource_usage']].rename(
        columns={'time': 'ds', 'resource_usage': 'y'})
else:
    raise ValueError("Required columns for resource usage not found")

if 'time' in df.columns and 'utility_allocation' in df.columns:
    df_utility = df[['time', 'utility_allocation']].rename(
        columns={'time': 'ds', 'utility_allocation': 'y'})
else:
    raise ValueError("Required columns for utility allocation not found")

# Step 5: Define holidays
print("\nStep 5: Defining holidays...")
if 'festive_season' in df.columns and 'time' in df.columns:
    festive_days = df[df['festive_season'] == 'Yes']['time'].dt.date.unique()
    holidays = pd.DataFrame({
        'holiday': 'festive_season',
        'ds': pd.to_datetime(festive_days),
        'lower_window': 0,
        'upper_window': 1,
    })
else:
    holidays = None
    print("Warning: Could not create holidays dataframe")

# Step 6-9: Build and train models
print("\nBuilding and training models...")


def build_model(df_train, holidays, regressor_cols):
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        seasonality_mode='multiplicative'
    )
    for regressor in regressor_cols:
        if regressor in df.columns:
            df_train[regressor] = df[regressor].values
            model.add_regressor(regressor)
    model.fit(df_train)
    return model


regressor_cols = ['days_to_deadline', 'shelf_life', 'department_encoded',
                  'priority_encoded', 'return_status_encoded', 'status_encoded']

model_resource = build_model(df_resource, holidays, regressor_cols)
model_utility = build_model(df_utility, holidays, regressor_cols)

# Make future dataframes
future = model_resource.make_future_dataframe(periods=7*24, freq='H')
for regressor in regressor_cols:
    if regressor in df.columns:
        future[regressor] = df[regressor].mean()

forecast_resource = model_resource.predict(future)
forecast_utility = model_utility.predict(future)

# Step 10: Visualize results
print("\nVisualizing results...")
fig1 = model_resource.plot(forecast_resource)
plt.title('Resource Usage Forecast')
plt.savefig('resource_forecast.png')

fig2 = model_utility.plot(forecast_utility)
plt.title('Utility Allocation Forecast')
plt.savefig('utility_forecast.png')

plot_components(model_resource, forecast_resource)
plt.savefig('resource_components.png')

plot_components(model_utility, forecast_utility)
plt.savefig('utility_components.png')

# Step 11: Model validation
print("\nModel validation...")


def evaluate_model(model, df_train):
    try:
        cv = cross_validation(model, initial='2 days',
                              period='1 day', horizon='1 day')
        return performance_metrics(cv)
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        return None


resource_metrics = evaluate_model(model_resource, df_resource)
utility_metrics = evaluate_model(model_utility, df_utility)

if resource_metrics is not None:
    print("\nResource Model Metrics:")
    print(resource_metrics.tail())
else:
    print("\nCould not compute resource model metrics")

if utility_metrics is not None:
    print("\nUtility Model Metrics:")
    print(utility_metrics.tail())
else:
    print("\nCould not compute utility model metrics")

# Save forecasts
forecast_resource[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    'resource_forecast.csv', index=False)
forecast_utility[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    'utility_forecast.csv', index=False)

# Save models
with open('resource_model.pkl', 'wb') as f:
    pickle.dump(model_resource, f)
with open('utility_model.pkl', 'wb') as f:
    pickle.dump(model_utility, f)

print("\nAll steps completed successfully!")

# Demand Forecasting Module


class DemandForecaster:
    """
    A class for predicting and forecasting supply chain demand using LSTM
    """

    def __init__(self, data_path):
        """
        Initialize the forecaster with data path

        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing supply chain data
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.features = None
        self.lookback = 3  # Default lookback window

    def load_data(self):
        """Load the supply chain data from CSV"""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess_data(self):
        """Preprocess the data for LSTM model"""
        if self.data is None:
            self.load_data()

        # Make a copy to avoid modifying the original data
        df = self.data.copy()

        # Convert date-time columns to datetime format
        df['time'] = pd.to_datetime(df['time'])
        df['deadline'] = pd.to_datetime(df['deadline'])

        # Extract features from datetime
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek

        # Calculate days until deadline
        df['days_to_deadline'] = (df['deadline'] - df['time']).dt.days

        # Encode categorical variables
        categorical_cols = ['department', 'priority', 'departure_loc',
                            'arrival_loc', 'return_status', 'festive_season', 'status']

        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        self.processed_data = df

        return self.processed_data

    def create_sequences(self, features, target_cols, lookback=None):
        """
        Create sequences for LSTM model

        Parameters:
        -----------
        features : list
            List of feature column names
        target_cols : list
            List of target column names to predict
        lookback : int, optional
            Number of time steps to look back

        Returns:
        --------
        X, y : numpy.ndarray
            Sequences and target values
        """
        if lookback is not None:
            self.lookback = lookback

        if self.processed_data is None:
            self.preprocess_data()

        data = self.processed_data

        # Scale the features
        self.scaler = MinMaxScaler()
        data[features] = self.scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[features].iloc[i:i+self.lookback].values)
            y.append(data[target_cols].iloc[i+self.lookback].values)

        self.features = features
        return np.array(X), np.array(y)

    def build_model(self, input_shape, output_dim):
        """
        Build an LSTM model

        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (lookback, n_features)
        output_dim : int
            Number of output dimensions

        Returns:
        --------
        model : keras.Model
            Compiled LSTM model
        """
        model = Sequential()
        model.add(LSTM(64, activation='relu',
                  input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    def train(self, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2):
        """
        Train the LSTM model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training sequences
        y_train : numpy.ndarray
            Target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Portion of training data to use for validation

        Returns:
        --------
        history : keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.model = self.build_model(
                (X_train.shape[1], X_train.shape[2]), y_train.shape[1])

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        return history

    def predict_next(self, sequence, target_cols):
        """
        Predict the next value given a sequence

        Parameters:
        -----------
        sequence : numpy.ndarray
            Input sequence
        target_cols : list
            List of target column names

        Returns:
        --------
        next_values : dict
            Dictionary of predicted values
        """
        # Reshape the sequence for prediction
        sequence_reshaped = sequence.reshape(
            1, sequence.shape[0], sequence.shape[1])

        # Make prediction
        prediction = self.model.predict(sequence_reshaped)[0]

        # Inverse transform for interpretability
        temp_df = pd.DataFrame(
            np.zeros((1, len(self.features))), columns=self.features)

        for i, col in enumerate(target_cols):
            temp_df[col] = prediction[i]

        prediction_original = self.scaler.inverse_transform(temp_df)

        # Extract predicted values
        next_values = {}
        for i, col in enumerate(target_cols):
            col_idx = self.features.index(col)
            next_values[col] = prediction_original[0, col_idx]

        return next_values

    def forecast(self, sequence, target_cols, n_steps=7):
        """
        Forecast future values

        Parameters:
        -----------
        sequence : numpy.ndarray
            Input sequence
        target_cols : list
            List of target column names
        n_steps : int
            Number of steps to forecast

        Returns:
        --------
        forecasts : list
            List of forecasted values
        """
        forecasts = []
        current_sequence = sequence.copy()

        for _ in range(n_steps):
            # Predict the next value
            next_pred = self.predict_next(current_sequence, target_cols)
            forecasts.append(next_pred)

            # Create a new data point with the predicted values
            new_point = current_sequence[-1].copy()

            # Update the target values in the normalized scale
            temp_df = pd.DataFrame(
                np.zeros((1, len(self.features))), columns=self.features)
            temp_df.iloc[0] = new_point
            temp_df_original = self.scaler.inverse_transform(temp_df)

            # Set the predicted values
            for col in target_cols:
                col_idx = self.features.index(col)
                temp_df_original[0, col_idx] = next_pred[col]

            # Transform back to normalized scale
            new_point_normalized = self.scaler.transform(temp_df_original)[0]

            # Update the sequence by removing the first element and adding the new prediction
            current_sequence = np.vstack(
                [current_sequence[1:], new_point_normalized])

        return forecasts

    def evaluate(self, X_test, y_test, target_cols):
        """
        Evaluate the model performance

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test sequences
        y_test : numpy.ndarray
            Actual target values
        target_cols : list
            List of target column names

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X_test)

        metrics = {}
        for i, col in enumerate(target_cols):
            mae = np.mean(np.abs(y_pred[:, i] - y_test[:, i]))
            mse = np.mean((y_pred[:, i] - y_test[:, i])**2)
            rmse = np.sqrt(mse)

            metrics[col] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }

        return metrics

    def plot_forecast(self, forecasts, target_cols, start_date=None):
        """
        Plot forecasted values

        Parameters:
        -----------
        forecasts : list
            List of forecasted values
        target_cols : list
            List of target column names
        start_date : datetime, optional
            Starting date for the forecast

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Matplotlib figure
        """
        if start_date is None:
            start_date = pd.Timestamp.now()

        # Create dates for the forecast period
        dates = [start_date + pd.Timedelta(days=i)
                 for i in range(len(forecasts))]

        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each target variable
        for col in target_cols:
            values = [forecast[col] for forecast in forecasts]
            ax.plot(dates, values, marker='o', label=col)

        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Forecast')
        ax.legend()
        ax.grid(True)

        return fig


# Example usage
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = DemandForecaster(
        'DemandSupply/data/ecommerce_supply_chain_data.csv')

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

    # Create sequences
    X, y = forecaster.create_sequences(features, target_cols, lookback=3)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build and train the model
    history = forecaster.train(X_train, y_train, epochs=50)

    # Evaluate the model
    metrics = forecaster.evaluate(X_test, y_test, target_cols)
    print("Evaluation Metrics:")
    for col in metrics:
        print(f"{col}:")
        for metric, value in metrics[col].items():
            print(f"  {metric}: {value:.4f}")

    # Generate a forecast
    sequence = X_test[0]  # Use the first test sequence for forecasting
    forecasts = forecaster.forecast(sequence, target_cols, n_steps=7)

    # Print forecasts
    print("\nForecasts:")
    for i, forecast in enumerate(forecasts):
        print(f"Day {i+1}:")
        for col, value in forecast.items():
            print(f"  {col}: {value:.2f}")

    # Plot the forecast
    fig = forecaster.plot_forecast(forecasts, target_cols)
    plt.show()
