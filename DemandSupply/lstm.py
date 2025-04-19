import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

# Load the data
data = pd.read_csv('supply_chain_data.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows of the dataset:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nSummary statistics:")
print(data.describe())

# Preprocessing
# Convert date-time to datetime format with correct format (DD-MM-YYYY)
data['time'] = pd.to_datetime(
    data['time'], format='%d-%m-%Y %H:%M', dayfirst=True)
data['deadline'] = pd.to_datetime(
    data['deadline'], format='%d-%m-%Y', dayfirst=True)

# Extract features from datetime
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['day_of_week'] = data['time'].dt.dayofweek

# Calculate days until deadline
data['days_to_deadline'] = (data['deadline'] - data['time']).dt.days

# Encode categorical variables
categorical_cols = ['department', 'priority', 'departure_loc',
                    'arrival_loc', 'return_status', 'festive_season', 'status']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col + '_encoded'] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features for the model
features = ['resource_usage', 'utility_allocation', 'shelf_life', 'hour', 'day', 'month',
            'day_of_week', 'days_to_deadline', 'department_encoded', 'priority_encoded',
            'departure_loc_encoded', 'arrival_loc_encoded', 'return_status_encoded',
            'festive_season_encoded', 'status_encoded']

# Scale the numerical features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Define the lookback window (number of previous time steps to use for prediction)
lookback = 3

# Function to create sequences for LSTM


def create_sequences(data, features, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[features].iloc[i:i+lookback].values)
        # Predict resource_usage and utility_allocation for the next time step
        y.append(data[['resource_usage', 'utility_allocation']
                      ].iloc[i+lookback].values)
    return np.array(X), np.array(y)


# Create sequences
X, y = create_sequences(data, features, lookback)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(
    lookback, len(features)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))  # Output resource_usage and utility_allocation predictions

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test MAE: {mae}')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the scaled values for better interpretability
# First, create a temporary DataFrame with zeros for all features
temp_df = pd.DataFrame(
    np.zeros((len(predictions), len(features))), columns=features)
# Set the predicted values for resource_usage and utility_allocation
temp_df['resource_usage'] = predictions[:, 0]
temp_df['utility_allocation'] = predictions[:, 1]
# Inverse transform
predictions_original_scale = scaler.inverse_transform(temp_df)[:, :2]

# Do the same for actual values
temp_df = pd.DataFrame(
    np.zeros((len(y_test), len(features))), columns=features)
temp_df['resource_usage'] = y_test[:, 0]
temp_df['utility_allocation'] = y_test[:, 1]
actual_original_scale = scaler.inverse_transform(temp_df)[:, :2]

# Plot the predictions vs actual values
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(actual_original_scale[:, 0],
            predictions_original_scale[:, 0], alpha=0.7)
plt.plot([actual_original_scale[:, 0].min(), actual_original_scale[:, 0].max()],
         [actual_original_scale[:, 0].min(), actual_original_scale[:, 0].max()], 'r--')
plt.xlabel('Actual Resource Usage')
plt.ylabel('Predicted Resource Usage')
plt.title('Resource Usage: Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(actual_original_scale[:, 1],
            predictions_original_scale[:, 1], alpha=0.7)
plt.plot([actual_original_scale[:, 1].min(), actual_original_scale[:, 1].max()],
         [actual_original_scale[:, 1].min(), actual_original_scale[:, 1].max()], 'r--')
plt.xlabel('Actual Utility Allocation')
plt.ylabel('Predicted Utility Allocation')
plt.title('Utility Allocation: Predictions vs Actual')
plt.tight_layout()
plt.show()

# Function to predict the next value given the current state


def predict_next(model, current_data, scaler, features, lookback):
    # Reshape data for LSTM input
    current_data_reshaped = current_data.reshape(1, lookback, len(features))

    # Make prediction
    prediction = model.predict(current_data_reshaped)[0]

    # Inverse transform for interpretability
    temp_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    temp_df['resource_usage'] = prediction[0]
    temp_df['utility_allocation'] = prediction[1]
    prediction_original = scaler.inverse_transform(temp_df)[0, :2]

    return prediction_original


# Example: Predict the next value for the last sequence in the test set
last_sequence = X_test[-1]
next_prediction = predict_next(
    model, last_sequence, scaler, features, lookback)
print(f"Predicted next resource usage: {next_prediction[0]:.2f}")
print(f"Predicted next utility allocation: {next_prediction[1]:.2f}")

# Feature importance analysis using a simple approach - correlation with prediction error


def feature_importance_analysis(model, X_test, y_test, features):
    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate absolute errors
    abs_errors = np.abs(y_pred - y_test)

    # Calculate mean error for each sample
    mean_errors = np.mean(abs_errors, axis=1)

    # For each feature, calculate correlation with prediction error
    importance_scores = []

    for i in range(len(features)):
        # Extract the feature values from the last time step of each sequence
        feature_values = X_test[:, -1, i]

        # Calculate correlation coefficient
        correlation = np.corrcoef(feature_values, mean_errors)[0, 1]
        importance_scores.append(abs(correlation))

    # Create a DataFrame to display feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    })

    return feature_importance.sort_values('Importance', ascending=False)


# Analyze feature importance
feature_importance = feature_importance_analysis(
    model, X_test, y_test, features)
print("\nFeature Importance Analysis:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score (Absolute Correlation with Error)')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Save the model
model.save('supply_chain_lstm_model.h5')
print("Model saved as 'supply_chain_lstm_model.h5'")

# Function to use the trained model for forecasting future values


def forecast_future(model, last_sequence, scaler, features, n_steps=7):
    """
    Forecast future values using the trained LSTM model

    Parameters:
    - model: Trained LSTM model
    - last_sequence: The last known sequence of data
    - scaler: The fitted scaler used to normalize the data
    - features: List of feature names
    - n_steps: Number of future steps to predict

    Returns:
    - forecasts: Array of predictions
    """
    forecasts = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        # Predict the next value
        next_pred = predict_next(
            model, current_sequence, scaler, features, lookback)
        forecasts.append(next_pred)

        # Create a new data point with the predicted values
        new_point = current_sequence[-1].copy()
        # Update resource_usage and utility_allocation in the normalized scale
        temp_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        temp_df.iloc[0] = new_point
        temp_df_original = scaler.inverse_transform(temp_df)

        # Set the predicted values
        temp_df_original[0, features.index('resource_usage')] = next_pred[0]
        temp_df_original[0, features.index(
            'utility_allocation')] = next_pred[1]

        # Transform back to normalized scale
        new_point_normalized = scaler.transform(temp_df_original)[0]

        # Update the sequence by removing the first element and adding the new prediction
        current_sequence = np.vstack(
            [current_sequence[1:], new_point_normalized])

    return np.array(forecasts)


# Example: Forecast the next 7 days using the last sequence in the dataset
future_forecast = forecast_future(
    model, X_test[-1], scaler, features, n_steps=7)
print("\nForecast for the next 7 days:")
for i, forecast in enumerate(future_forecast):
    print(
        f"Day {i+1}: Resource Usage = {forecast[0]:.2f}, Utility Allocation = {forecast[1]:.2f}")

# Function to save the data to csv


def save_to_csv(file_path, original_data, data_type):
    """
    Save data to CSV file

    Parameters:
    - file_path: File path to save the CSV
    - original_data: Data to save
    - data_type: Type of data (predictions, actual, etc.)
    """
    df = pd.DataFrame(original_data, columns=[
                      'resource_usage', 'utility_allocation'])
    df['type'] = data_type
    df.to_csv(file_path, index=False)
    print(f"Saved {data_type} to {file_path}")


# Save predictions and actual values to CSV for further analysis
save_to_csv('lstm_predictions.csv', predictions_original_scale, 'predictions')
save_to_csv('lstm_actual_values.csv', actual_original_scale, 'actual')
save_to_csv('lstm_future_forecast.csv', future_forecast, 'forecast')
