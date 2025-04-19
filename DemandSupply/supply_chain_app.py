import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
import os
import io
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the data paths using os.path for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try different path patterns to locate the data file
possible_data_paths = [
    os.path.join(BASE_DIR, 'data', 'ecommerce_supply_chain_data.csv'),
    os.path.join(os.path.dirname(BASE_DIR), 'data',
                 'ecommerce_supply_chain_data.csv'),
    os.path.join(BASE_DIR, 'DemandSupply', 'data',
                 'ecommerce_supply_chain_data.csv'),
    os.path.join('V:', 'Vaibhav', 'cod3', 'SupplyChainOptimizer_Neuvera',
                 'DemandSupply', 'data', 'ecommerce_supply_chain_data.csv'),
    '/v:/Vaibhav/cod3/SupplyChainOptimizer_Neuvera/DemandSupply/data/ecommerce_supply_chain_data.csv',
]

# Define possible model paths
MODEL_PATHS = [
    os.path.join(BASE_DIR, 'models', 'supply_chain_lstm_model.h5'),
    os.path.join(os.path.dirname(BASE_DIR), 'models',
                 'supply_chain_lstm_model.h5'),
    os.path.join(BASE_DIR, 'DemandSupply', 'models',
                 'supply_chain_lstm_model.h5'),
    os.path.join('V:', 'Vaibhav', 'cod3', 'SupplyChainOptimizer_Neuvera',
                 'DemandSupply', 'models', 'supply_chain_lstm_model.h5'),
    '/v:/Vaibhav/cod3/SupplyChainOptimizer_Neuvera/DemandSupply/models/supply_chain_lstm_model.h5',
]
MODEL_PATH = MODEL_PATHS[0]  # Use the first one as default

# Function to ensure model directory exists


def ensure_model_dir():
    model_dir = os.path.dirname(MODEL_PATH)
    try:
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    except:
        # Fallback - create directory in the current working directory
        os.makedirs('models', exist_ok=True)
        return 'models'

# Function to load the data


def load_data():
    # Try each possible path until we find the data file
    for path in possible_data_paths:
        try:
            st.write(f"Attempting to load data from: {path}")
            data = pd.read_csv(path)
            st.success(f"Data loaded successfully from: {path}")
            return data
        except Exception as e:
            continue

    # If we get here, none of the paths worked
    st.error(
        f"Could not load data from any of the attempted paths. Please verify the file exists at one of these locations: {possible_data_paths}")

    # Offer a fallback option
    st.write("As a fallback option, you can upload it directly: ")
    uploaded_file = st.file_uploader(
        "Upload ecommerce_supply_chain_data.csv", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            return data
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    return None

# Function to preprocess the data


def preprocess_data(data):
    # Make a copy to avoid modifying the original data
    df = data.copy()

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
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

# Function to create sequences for LSTM


def create_sequences(data, features, target_cols, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[features].iloc[i:i+lookback].values)
        y.append(data[target_cols].iloc[i+lookback].values)
    return np.array(X), np.array(y)

# Modified function to build and train the LSTM model


def build_train_model(X_train, y_train, lookback, n_features, n_outputs, epochs=100):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(
        lookback, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs))

    # Use explicit metric instances to avoid serialization issues
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[MeanSquaredError(), MeanAbsoluteError()])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history

# Function to save model and metadata


def save_model_with_metadata(model, model_path, features, target_cols, lookback, scaler):
    # Save the model
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Save the Keras model
    model.save(model_path, save_format='h5')

    # Save the features and target columns
    np.save(os.path.join(model_dir, 'features.npy'), features)
    np.save(os.path.join(model_dir, 'target_cols.npy'), target_cols)

    # Save the lookback value
    with open(os.path.join(model_dir, 'lookback.json'), 'w') as f:
        json.dump({'lookback': lookback}, f)

    # Save the scaler to pickle file
    import pickle
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

# Modified function to load model with error handling for metrics and scaler issues


def load_model_with_metadata(model_path):
    try:
        # First attempt to load the model normally
        model = load_model(model_path, compile=True)
    except Exception as e:
        # If there's an error regarding metrics, try loading with custom objects
        if "Could not locate function 'mse'" in str(e):
            # Define custom objects to handle the metrics issue
            custom_objects = {
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
            }
            try:
                # Try loading with custom objects
                model = load_model(
                    model_path, custom_objects=custom_objects, compile=True)
            except Exception as load_error:
                # If that still fails, try loading without compiling then recompile
                try:
                    model = load_model(model_path, compile=False)
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=[tf.keras.metrics.MeanSquaredError(),
                                 tf.keras.metrics.MeanAbsoluteError()]
                    )
                except Exception as final_error:
                    # If all else fails, raise a more detailed error
                    raise ValueError(
                        f"Failed to load model due to metric error. Complete error: {final_error}") from final_error
        else:
            # If it's not a metric error, re-raise
            raise

    # Load features and target columns
    model_dir = os.path.dirname(model_path)

    try:
        features = np.load(os.path.join(
            model_dir, 'features.npy'), allow_pickle=True)
        target_cols = np.load(os.path.join(
            model_dir, 'target_cols.npy'), allow_pickle=True)
    except Exception as e:
        # If we can't load features/target_cols, use defaults
        st.warning(
            f"Could not load model metadata. Using default values. Error: {e}")
        features = [
            'resource_usage', 'utility_allocation', 'shelf_life', 'hour', 'day', 'month',
            'day_of_week', 'days_to_deadline', 'department_encoded', 'priority_encoded',
            'departure_loc_encoded', 'arrival_loc_encoded', 'return_status_encoded',
            'festive_season_encoded', 'status_encoded'
        ]
        target_cols = ["resource_usage", "utility_allocation"]

    # Try to load lookback, default to 3 if not found
    try:
        with open(os.path.join(model_dir, 'lookback.json'), 'r') as f:
            lookback_data = json.load(f)
            lookback = lookback_data['lookback']
    except:
        lookback = 3  # Default lookback

    # Try to load scaler, create a new one if not found
    scaler = None
    try:
        import pickle
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        # Test if the scaler is fitted by trying to transform a sample
        test_data = np.zeros((1, len(features)))
        try:
            scaler.transform(test_data)
        except Exception as scaler_error:
            if "is not fitted yet" in str(scaler_error):
                st.warning(
                    "Loaded scaler is not fitted. Will create a new one.")
                scaler = None
    except Exception as e:
        st.warning(f"Could not load scaler: {e}. Will create a new one.")

    # If scaler is None, we need to create and fit a new one
    if scaler is None:
        st.warning("Creating and fitting a new scaler with the available data.")
        scaler = MinMaxScaler()

    return model, features, target_cols, lookback, scaler

# Function to predict the next value given the current state


def predict_next(model, current_data, scaler, features, target_cols, lookback):
    # Reshape data for LSTM inputs
    current_data_reshaped = current_data.reshape(1, lookback, len(features))

    # Make prediction
    prediction = model.predict(current_data_reshaped)[0]

    # Inverse transform for interpretability
    temp_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    for i, col in enumerate(target_cols):
        temp_df[col] = prediction[i]

    prediction_original = scaler.inverse_transform(temp_df)
    prediction_values = {}

    for i, col in enumerate(target_cols):
        col_idx = features.index(col)
        prediction_values[col] = prediction_original[0, col_idx]

    return prediction_values

# Function to forecast future values


def forecast_future(model, last_sequence, scaler, features, target_cols, lookback, n_steps=7):
    forecasts = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        # Predict the next value
        next_pred = predict_next(
            model, current_sequence, scaler, features, target_cols, lookback)
        forecasts.append(next_pred)

        # Create a new data point with the predicted values
        new_point = current_sequence[-1].copy()

        # Update the target values in the normalized scale
        temp_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        temp_df.iloc[0] = new_point
        temp_df_original = scaler.inverse_transform(temp_df)

        # Set the predicted values
        for col in target_cols:
            col_idx = features.index(col)
            temp_df_original[0, col_idx] = next_pred[col]

        # Transform back to normalized scale
        new_point_normalized = scaler.transform(temp_df_original)[0]

        # Update the sequence by removing the first element and adding the new prediction
        current_sequence = np.vstack(
            [current_sequence[1:], new_point_normalized])

    return forecasts

# Corrected function to calculate estimated stock levels based on forecast


def calculate_stock_levels(forecast_df, initial_stock, restock_rate):
    # Copy the forecast dataframe
    stock_df = forecast_df.copy()

    # Add stock level column initialized with initial stock value
    stock_df['stock_level'] = initial_stock

    # Calculate daily stock consumption based on resource usage
    # Assuming resource usage is proportional to stock consumption
    stock_df['consumption'] = stock_df['resource_usage'] * 0.1

    # Calculate daily stock levels
    for i in range(1, len(stock_df)):
        # Previous stock minus consumption plus restocking
        stock_df.loc[i, 'stock_level'] = max(0,
                                             stock_df.loc[i-1, 'stock_level'] -
                                             stock_df.loc[i-1, 'consumption'] +
                                             restock_rate)

    return stock_df

# Function to save chart as an image


def get_chart_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Main app function


def main():
    # Sidebar with app information
    st.sidebar.title("📊 Supply Chain Analytics")
    st.sidebar.info(
        "This application uses LSTM neural networks to predict and forecast "
        "supply chain demand based on historical data."
    )

    # Add tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration",
        "🧮 Model Training",
        "📈 Forecasting",
        "🔍 Insights"
    ])

    # Load the data
    data = load_data()

    if data is None:
        st.error("Failed to load data. Please check the file path.")
        return

    # DATA EXPLORATION TAB
    with tab1:
        st.header("Data Exploration")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("### Dataset Overview")
            st.dataframe(data.head(10))

        with col2:
            st.write("### Dataset Information")
            st.write(f"**Number of Records:** {data.shape[0]}")
            st.write(f"**Number of Features:** {data.shape[1]}")

            # Missing values info
            missing_values = data.isnull().sum().sum()
            st.write(f"**Missing Values:** {missing_values}")

            # Display data types counts
            dtype_counts = data.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            st.write("**Data Types:**")
            st.dataframe(dtype_counts)

        # Data visualization section
        st.write("### Data Visualization")

        # Allow user to select visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Resource Usage by Department", "Utility Allocation by Priority",
             "Resource Usage vs Utility Allocation", "Time Series Analysis",
             "Department Distribution"]
        )

        if viz_type == "Resource Usage by Department":
            fig = px.box(data, x="department", y="resource_usage",
                         color="department", title="Resource Usage by Department")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Utility Allocation by Priority":
            fig = px.box(data, x="priority", y="utility_allocation",
                         color="priority", title="Utility Allocation by Priority")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Resource Usage vs Utility Allocation":
            fig = px.scatter(data, x="resource_usage", y="utility_allocation",
                             color="department", hover_data=["priority", "status"],
                             title="Resource Usage vs Utility Allocation")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Time Series Analysis":
            data_sorted = data.copy()
            data_sorted['time'] = pd.to_datetime(data_sorted['time'])
            data_sorted = data_sorted.sort_values('time')

            # Resample by day for visualization
            daily_data = data_sorted.set_index('time')
            daily_resource = daily_data['resource_usage'].resample('D').mean()
            daily_utility = daily_data['utility_allocation'].resample(
                'D').mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_resource.index, y=daily_resource.values,
                                     mode='lines', name='Resource Usage'))
            fig.add_trace(go.Scatter(x=daily_utility.index, y=daily_utility.values,
                                     mode='lines', name='Utility Allocation'))
            fig.update_layout(title='Resource Usage and Utility Allocation Over Time',
                              xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Department Distribution":
            dept_counts = data['department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            fig = px.pie(dept_counts, values='Count', names='Department',
                         title='Distribution of Departments')
            st.plotly_chart(fig, use_container_width=True)

    # MODEL TRAINING TAB
    with tab2:
        st.header("Model Training and Evaluation")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Model parameters selection
            st.write("### Model Parameters")
            lookback = st.slider("Lookback Window Size", min_value=1, max_value=10, value=3,
                                 help="Number of time steps to look back for making predictions")

            epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=100,
                               help="Maximum number of training epochs (early stopping may terminate sooner)")

            target_options = ["resource_usage",
                              "utility_allocation", "bottleneck_flag"]
            target_cols = st.multiselect(
                "Select Target Variables to Predict",
                options=target_options,
                default=["resource_usage", "utility_allocation"],
                help="Variables to predict with the model"
            )

            if not target_cols:
                st.warning("Please select at least one target variable")
                target_cols = ["resource_usage"]

        with col2:
            st.write("### Training Configuration")
            train_size = st.slider("Training Data Percentage", min_value=50, max_value=90, value=80,
                                   help="Percentage of data to use for training") / 100

            validation_split = st.slider("Validation Split", min_value=10, max_value=30, value=20,
                                         help="Percentage of training data to use for validation") / 100

            use_early_stopping = st.checkbox("Use Early Stopping", value=True,
                                             help="Stop training when validation loss stops improving")

            train_button = st.button("Train Model", type="primary")

        # Preprocess data and train model when button is clicked
        if train_button:
            with st.spinner("Preprocessing data and training model..."):
                # Preprocess the data
                processed_data, label_encoders = preprocess_data(data)

                # Define features
                features = [
                    'resource_usage', 'utility_allocation', 'shelf_life', 'hour', 'day', 'month',
                    'day_of_week', 'days_to_deadline', 'department_encoded', 'priority_encoded',
                    'departure_loc_encoded', 'arrival_loc_encoded', 'return_status_encoded',
                    'festive_season_encoded', 'status_encoded'
                ]

                # Scale the features
                scaler = MinMaxScaler()
                processed_data[features] = scaler.fit_transform(
                    processed_data[features])

                # Create sequences
                X, y = create_sequences(
                    processed_data, features, target_cols, lookback)

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1-train_size, random_state=42
                )

                # Build and train the model
                model, history = build_train_model(
                    X_train, y_train, lookback, len(
                        features), len(target_cols), epochs
                )

                # Save the model and related artifacts
                model_dir = ensure_model_dir()
                model_save_path = os.path.join(
                    model_dir, 'supply_chain_lstm_model.h5')

                # Use the new save function instead
                save_model_with_metadata(
                    model, model_save_path, features, target_cols, lookback, scaler)

                st.success(f"Model trained and saved to: {model_save_path}")

                # Display training history
                st.write("### Training History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))),
                                         y=history.history['loss'],
                                         mode='lines', name='Training Loss'))
                fig.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))),
                                         y=history.history['val_loss'],
                                         mode='lines', name='Validation Loss'))
                fig.update_layout(title='Training and Validation Loss',
                                  xaxis_title='Epoch',
                                  yaxis_title='Loss')
                st.plotly_chart(fig, use_container_width=True)

                # Model evaluation
                st.write("### Model Evaluation")
                y_pred = model.predict(X_test)

                eval_metrics = {}
                for i, col in enumerate(target_cols):
                    mae = np.mean(np.abs(y_pred[:, i] - y_test[:, i]))
                    mse = np.mean((y_pred[:, i] - y_test[:, i])**2)
                    rmse = np.sqrt(mse)

                    eval_metrics[col] = {
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse
                    }

                # Display metrics
                metrics_df = pd.DataFrame({
                    "Target": list(eval_metrics.keys()),
                    "MAE": [eval_metrics[k]["MAE"] for k in eval_metrics],
                    "MSE": [eval_metrics[k]["MSE"] for k in eval_metrics],
                    "RMSE": [eval_metrics[k]["RMSE"] for k in eval_metrics]
                })

                st.dataframe(metrics_df)

                # Plot actual vs. predicted
                st.write("### Actual vs. Predicted Values")

                for i, col in enumerate(target_cols):
                    fig = px.scatter(x=y_test[:, i], y=y_pred[:, i],
                                     labels={'x': f'Actual {col}',
                                             'y': f'Predicted {col}'},
                                     title=f'Actual vs. Predicted {col}')

                    # Add 45-degree line
                    min_val = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
                    max_val = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                             mode='lines', name='Ideal',
                                             line=dict(color='red', dash='dash')))

                    st.plotly_chart(fig, use_container_width=True)

                # Feature importance (using permutation importance for simplicity)
                st.write("### Feature Importance")

                # Simple correlation-based feature importance
                importance_scores = []

                for i in range(len(features)):
                    feature_values = X_test[:, -1, i]

                    # Calculate mean absolute error for all target variables
                    mean_errors = np.mean([np.abs(y_pred[:, j] - y_test[:, j])
                                          for j in range(len(target_cols))], axis=0)

                    # Calculate correlation coefficient
                    correlation = np.corrcoef(
                        feature_values, mean_errors)[0, 1]
                    importance_scores.append(abs(correlation))

                # Create a DataFrame to display feature importance
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False)

                fig = px.bar(feature_importance.head(10), x='Importance', y='Feature',
                             orientation='h', title='Top 10 Features by Importance')
                st.plotly_chart(fig, use_container_width=True)

    # FORECASTING TAB
    with tab3:
        st.header("Demand Forecasting & Stock Optimization")

        # Check if model exists in any of the possible locations
        model_exists = any(os.path.exists(path) for path in MODEL_PATHS)

        if not model_exists:
            st.warning(
                "No trained model found. Please train a model in the 'Model Training' tab first.")
        else:
            # Load the model and related artifacts
            try:
                # Find which model path exists
                model_path = next(
                    (path for path in MODEL_PATHS if os.path.exists(path)), None)

                # Use the new load function
                with st.spinner("Loading model..."):
                    model, features, target_cols, lookback, scaler = load_model_with_metadata(
                        model_path)

                st.success(f"Model loaded successfully from: {model_path}")

                # Create two sections with tabs
                forecast_tab1, forecast_tab2 = st.tabs(
                    ["Standard Forecast", "Scenario-Based Forecast"])

                # STANDARD FORECAST TAB
                with forecast_tab1:
                    st.write("### Standard Forecast Configuration")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        forecast_days = st.slider("Number of Days to Forecast",
                                                  min_value=1, max_value=30, value=7,
                                                  help="Number of future days to forecast")

                        # Let user select a department to filter data
                        departments = data['department'].unique()
                        selected_dept = st.selectbox("Select Department for Forecasting",
                                                     departments)

                        st.write(
                            f"Generating forecast for department: **{selected_dept}**")

                    with col2:
                        st.write("### Forecast Details")

                        today = datetime.now()
                        st.write(
                            f"**Current Date:** {today.strftime('%Y-%m-%d')}")

                        forecast_end = today + timedelta(days=forecast_days)
                        st.write(
                            f"**Forecast End Date:** {forecast_end.strftime('%Y-%m-%d')}")

                        forecast_button = st.button(
                            "Generate Forecast", type="primary")

                    if forecast_button:
                        with st.spinner("Generating forecast..."):
                            # Preprocess the data
                            processed_data, _ = preprocess_data(data)

                            # Filter data for the selected department
                            dept_data = processed_data[processed_data['department']
                                                       == selected_dept]

                            if len(dept_data) == 0:
                                st.error(
                                    f"No data available for department: {selected_dept}")
                            else:
                                try:
                                    # Check if scaler is fitted
                                    try:
                                        # Try to transform a sample row to check if scaler is fitted
                                        sample = dept_data[features].iloc[[0]]
                                        _ = scaler.transform(sample)
                                    except Exception as e:
                                        # If scaler is not fitted, fit it with all data first
                                        if "is not fitted yet" in str(e):
                                            st.info(
                                                "Fitting scaler with available data...")
                                            scaler.fit(
                                                processed_data[features])
                                        else:
                                            raise e

                                    # Now scale the department data
                                    dept_data[features] = scaler.transform(
                                        dept_data[features])

                                    # Create sequences for the department data
                                    if len(dept_data) <= lookback:
                                        st.error(
                                            f"Not enough data for department {selected_dept}. Need at least {lookback+1} records, but only have {len(dept_data)}.")
                                    else:
                                        X_dept, _ = create_sequences(
                                            dept_data, features, target_cols, lookback)

                                        # Use the last sequence for forecasting
                                        last_sequence = X_dept[-1]

                                        # Generate forecast
                                        forecasts = forecast_future(
                                            model, last_sequence, scaler, features, target_cols, lookback, forecast_days
                                        )

                                        # Create forecast DataFrame
                                        forecast_dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d')
                                                          for i in range(forecast_days)]

                                        forecast_data = {
                                            'Date': forecast_dates,
                                        }

                                        for col in target_cols:
                                            forecast_data[col] = [forecast[col]
                                                                  for forecast in forecasts]

                                        forecast_df = pd.DataFrame(
                                            forecast_data)

                                        # Display forecast
                                        st.write("### Forecast Results")
                                        st.dataframe(forecast_df.style.format({
                                            col: "{:.2f}" for col in target_cols
                                        }))

                                        # Plot forecast
                                        st.write("### Forecast Visualization")

                                        # Create interactive visualization with plotly
                                        fig = go.Figure()

                                        for col in target_cols:
                                            fig.add_trace(go.Scatter(
                                                x=forecast_df['Date'],
                                                y=forecast_df[col],
                                                mode='lines+markers',
                                                name=col
                                            ))

                                        fig.update_layout(
                                            title=f'Forecast for {selected_dept} Department',
                                            xaxis_title='Date',
                                            yaxis_title='Value',
                                            hovermode='x unified'
                                        )

                                        st.plotly_chart(
                                            fig, use_container_width=True)

                                        # Stock level simulation
                                        st.write("### Stock Level Simulation")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            initial_stock = st.slider("Initial Stock Level",
                                                                      min_value=100,
                                                                      max_value=1000,
                                                                      value=500,
                                                                      help="Starting inventory level")

                                        with col2:
                                            restock_rate = st.slider("Daily Restock Rate",
                                                                     min_value=0,
                                                                     max_value=100,
                                                                     value=20,
                                                                     help="Units added to inventory each day")

                                        # Calculate stock levels based on forecast
                                        try:
                                            stock_df = calculate_stock_levels(
                                                forecast_df, initial_stock, restock_rate)

                                            # Plot stock levels
                                            fig = go.Figure()

                                            fig.add_trace(go.Scatter(
                                                x=stock_df['Date'],
                                                y=stock_df['stock_level'],
                                                mode='lines+markers',
                                                name='Stock Level',
                                                line=dict(color='green')
                                            ))

                                            # Add a line for the safety stock (20% of initial)
                                            safety_stock = initial_stock * 0.2
                                            fig.add_trace(go.Scatter(
                                                x=stock_df['Date'],
                                                y=[safety_stock] *
                                                len(stock_df),
                                                mode='lines',
                                                name='Safety Stock',
                                                line=dict(
                                                    color='red', dash='dash')
                                            ))

                                            fig.update_layout(
                                                title=f'Projected Stock Levels for {selected_dept}',
                                                xaxis_title='Date',
                                                yaxis_title='Stock Level',
                                                hovermode='x unified'
                                            )

                                            st.plotly_chart(
                                                fig, use_container_width=True)

                                            # Generate and display optimization recommendations
                                            st.write(
                                                "### Stock Optimization Recommendations")

                                            # Function to generate optimization recommendations based on forecast
                                            def generate_optimization_recommendations(forecast_df, stock_levels, department, threshold=0.7):
                                                recommendations = []

                                                # Calculate average resource usage from forecast
                                                avg_resource_usage = forecast_df['resource_usage'].mean(
                                                )

                                                # Check if there's a high resource usage period
                                                high_usage_days = forecast_df[forecast_df['resource_usage']
                                                                              > avg_resource_usage * 1.2]
                                                if not high_usage_days.empty:
                                                    high_dates = ', '.join(
                                                        high_usage_days['Date'].iloc[:3].tolist())
                                                    recommendations.append(
                                                        f"⚠️ High resource usage expected on {high_dates}. Consider increasing stock levels before these dates.")

                                                # Check if stock levels are below threshold
                                                for product, level in stock_levels.items():
                                                    if level < threshold:
                                                        recommendations.append(
                                                            f"⚠️ {product} stock level is low ({level*100:.1f}%). Restock recommended.")

                                                # General recommendations
                                                recommendations.append(
                                                    f"📊 Optimize inventory for {department} based on forecasted resource usage pattern.")
                                                recommendations.append(
                                                    "🔄 Consider implementing just-in-time inventory for products with stable demand.")
                                                recommendations.append(
                                                    "📅 Schedule deliveries to align with forecasted demand peaks.")

                                                # If utility allocation is in forecast
                                                if 'utility_allocation' in forecast_df.columns:
                                                    # Calculate ratio between resource usage and utility allocation
                                                    forecast_df['efficiency'] = forecast_df['utility_allocation'] / \
                                                        forecast_df['resource_usage']
                                                    avg_efficiency = forecast_df['efficiency'].mean(
                                                    )

                                                    if avg_efficiency < 0.8:
                                                        recommendations.append(
                                                            "⚡ Current utility allocation efficiency is below optimal levels. Consider resource reallocation.")
                                                    else:
                                                        recommendations.append(
                                                            "✅ Utility allocation efficiency is at optimal levels.")

                                                return recommendations

                                            # Create simulated stock levels for multiple products
                                            stock_levels = {
                                                "Product A": stock_df['stock_level'].iloc[-1] / initial_stock,
                                                "Product B": np.random.uniform(0.3, 0.9),
                                                "Product C": np.random.uniform(0.5, 0.95)
                                            }

                                            recommendations = generate_optimization_recommendations(
                                                forecast_df, stock_levels, selected_dept)

                                            for rec in recommendations:
                                                st.write(f"- {rec}")

                                            # Allow the user to download the forecast
                                            csv = forecast_df.to_csv(
                                                index=False)
                                            b64 = base64.b64encode(
                                                csv.encode()).decode()
                                            href = f'<a href="data:file/csv;base64,{b64}" download="forecast_{selected_dept}.csv">Download Forecast CSV</a>'
                                            st.markdown(
                                                href, unsafe_allow_html=True)

                                        except Exception as stock_error:
                                            st.error(
                                                f"Error calculating stock levels: {stock_error}")
                                            st.info(
                                                "Try a different department or retrain the model.")

                                except Exception as e:
                                    st.error(f"Error generating forecast: {e}")
                                    st.info(
                                        "Try retraining the model to fix this issue.")

                # SCENARIO-BASED FORECAST TAB
                with forecast_tab2:
                    st.write("### Scenario-Based Forecast")
                    st.write(
                        "Create custom scenarios to evaluate different supply chain conditions.")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Scenario parameters
                        st.write("#### Scenario Parameters")

                        # Let user select a department for the scenario
                        departments = data['department'].unique()
                        selected_dept = st.selectbox(
                            "Select Department", departments, key="scenario_dept")

                        # Add scenario modifiers
                        resource_modifier = st.slider("Resource Usage Modifier",
                                                      min_value=0.5,
                                                      max_value=2.0,
                                                      value=1.0,
                                                      step=0.1,
                                                      help="Multiply forecasted resource usage by this factor")

                        utility_modifier = st.slider("Utility Allocation Modifier",
                                                     min_value=0.5,
                                                     max_value=2.0,
                                                     value=1.0,
                                                     step=0.1,
                                                     help="Multiply forecasted utility allocation by this factor")

                        forecast_days = st.slider("Forecast Days",
                                                  min_value=1,
                                                  max_value=30,
                                                  value=7,
                                                  key="scenario_days")

                    with col2:
                        st.write("#### Scenario Information")

                        if resource_modifier > 1:
                            st.warning(
                                f"Resource usage increased by {(resource_modifier-1)*100:.0f}%")
                        elif resource_modifier < 1:
                            st.info(
                                f"Resource usage decreased by {(1-resource_modifier)*100:.0f}%")

                        if utility_modifier > 1:
                            st.warning(
                                f"Utility allocation increased by {(utility_modifier-1)*100:.0f}%")
                        elif utility_modifier < 1:
                            st.info(
                                f"Utility allocation decreased by {(1-utility_modifier)*100:.0f}%")

                        scenario_button = st.button(
                            "Run Scenario Forecast", type="primary", key="run_scenario")

                    if scenario_button:
                        with st.spinner("Generating scenario forecast..."):
                            try:
                                # Process the data similar to the standard forecast
                                processed_data, _ = preprocess_data(data)

                                # Filter for selected department
                                dept_data = processed_data[processed_data['department']
                                                           == selected_dept]

                                if len(dept_data) == 0:
                                    st.error(
                                        f"No data available for department: {selected_dept}")
                                else:
                                    # Check and fit scaler if needed
                                    try:
                                        sample = dept_data[features].iloc[[0]]
                                        _ = scaler.transform(sample)
                                    except Exception as scaler_error:
                                        if "is not fitted yet" in str(scaler_error):
                                            st.info(
                                                "Fitting scaler with available data...")
                                            scaler.fit(
                                                processed_data[features])
                                        else:
                                            raise scaler_error

                                    # Scale department data
                                    dept_data[features] = scaler.transform(
                                        dept_data[features])

                                    # Create sequences
                                    if len(dept_data) <= lookback:
                                        st.error(
                                            f"Not enough data for department {selected_dept}. Need at least {lookback+1} records.")
                                    else:
                                        X_dept, _ = create_sequences(
                                            dept_data, features, target_cols, lookback)
                                        last_sequence = X_dept[-1]

                                        # Generate base forecast
                                        base_forecasts = forecast_future(
                                            model, last_sequence, scaler, features, target_cols, lookback, forecast_days
                                        )

                                        # Apply scenario modifiers
                                        scenario_forecasts = []
                                        for forecast in base_forecasts:
                                            modified_forecast = forecast.copy()

                                            if 'resource_usage' in modified_forecast:
                                                modified_forecast['resource_usage'] *= resource_modifier

                                            if 'utility_allocation' in modified_forecast:
                                                modified_forecast['utility_allocation'] *= utility_modifier

                                            scenario_forecasts.append(
                                                modified_forecast)

                                        # Create forecast DataFrame
                                        forecast_dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                                                          for i in range(forecast_days)]

                                        forecast_data = {
                                            'Date': forecast_dates,
                                        }

                                        for col in target_cols:
                                            if col in scenario_forecasts[0]:
                                                forecast_data[col] = [
                                                    forecast[col] for forecast in scenario_forecasts]

                                        scenario_df = pd.DataFrame(
                                            forecast_data)

                                        # Display forecast results
                                        st.write(
                                            "### Scenario Forecast Results")
                                        st.dataframe(scenario_df.style.format({
                                            col: "{:.2f}" for col in target_cols if col in scenario_df.columns
                                        }))

                                        # Plot scenario forecast
                                        st.write("### Scenario Visualization")
                                        fig = go.Figure()

                                        for col in target_cols:
                                            if col in scenario_df.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=scenario_df['Date'],
                                                    y=scenario_df[col],
                                                    mode='lines+markers',
                                                    name=f"{col} (Scenario)"
                                                ))

                                        fig.update_layout(
                                            title=f'Scenario Forecast for {selected_dept} Department',
                                            xaxis_title='Date',
                                            yaxis_title='Value',
                                            hovermode='x unified'
                                        )

                                        st.plotly_chart(
                                            fig, use_container_width=True)

                                        # Impact analysis
                                        st.write(
                                            "### Scenario Impact Analysis")

                                        # Calculate total resource usage and utility allocation
                                        total_resource = scenario_df['resource_usage'].sum(
                                        ) if 'resource_usage' in scenario_df.columns else 0
                                        total_utility = scenario_df['utility_allocation'].sum(
                                        ) if 'utility_allocation' in scenario_df.columns else 0

                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.metric(
                                                label="Total Resource Usage",
                                                value=f"{total_resource:.2f}",
                                                delta=f"{(resource_modifier-1)*100:.1f}% vs Baseline"
                                            )

                                        with col2:
                                            st.metric(
                                                label="Total Utility Allocation",
                                                value=f"{total_utility:.2f}",
                                                delta=f"{(utility_modifier-1)*100:.1f}% vs Baseline"
                                            )

                                        # Generate recommendations based on scenario
                                        st.write(
                                            "### Scenario Recommendations")

                                        recommendations = []

                                        if resource_modifier > 1.2:
                                            recommendations.append(
                                                "⚠️ High resource usage in this scenario may lead to bottlenecks. Consider increasing capacity.")
                                        if resource_modifier < 0.8:
                                            recommendations.append(
                                                "ℹ️ Lower resource usage suggests potential for cost savings or resource reallocation.")

                                        if utility_modifier > 1.2:
                                            recommendations.append(
                                                "⚠️ Increased utility allocation may impact profitability. Review cost structures.")
                                        if utility_modifier < 0.8:
                                            recommendations.append(
                                                "ℹ️ Reduced utility allocation may indicate efficiency improvements or underfunding.")

                                        if total_resource > 0 and total_utility > 0:
                                            efficiency = total_utility / total_resource
                                            if efficiency < 0.8:
                                                recommendations.append(
                                                    "⚠️ Low efficiency ratio detected. Review resource allocation strategy.")
                                            if efficiency > 1.2:
                                                recommendations.append(
                                                    "✅ High efficiency ratio suggests good resource utilization.")

                                        if not recommendations:
                                            recommendations.append(
                                                "✅ No significant issues detected in this scenario.")

                                        for rec in recommendations:
                                            st.write(f"- {rec}")

                                        # Allow download of scenario forecast
                                        csv = scenario_df.to_csv(index=False)
                                        b64 = base64.b64encode(
                                            csv.encode()).decode()
                                        href = f'<a href="data:file/csv;base64,{b64}" download="scenario_forecast_{selected_dept}.csv">Download Scenario Forecast CSV</a>'
                                        st.markdown(
                                            href, unsafe_allow_html=True)

                            except Exception as scenario_error:
                                st.error(
                                    f"Error generating scenario forecast: {scenario_error}")
                                st.info(
                                    "Try retraining the model to fix this issue.")

            except Exception as e:
                st.error(f"Error loading model or generating forecast: {e}")
                st.info(
                    "Try retraining the model on the 'Model Training' tab with the updated code to resolve this issue.")
                if "is not fitted yet" in str(e):
                    st.info(
                        "The scaler needs to be fitted to the data. Please retrain the model to fix this issue.")

    # INSIGHTS TAB
    with tab4:
        # Create sections for different types of insights
        st.header("Supply Chain Insights")

        # Create sections for different types of insights
        st.write("### Key Performance Indicators")

        # Calculate KPIs from the data
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            avg_resource = np.mean(data['resource_usage'])
            st.metric(label="Avg. Resource Usage", value=f"{avg_resource:.2f}")

        with kpi2:
            avg_utility = np.mean(data['utility_allocation'])
            st.metric(label="Avg. Utility Allocation",
                      value=f"{avg_utility:.2f}")

        with kpi3:
            bottleneck_pct = (data['bottleneck_flag'].sum() / len(data)) * 100
            st.metric(label="Bottleneck Percentage",
                      value=f"{bottleneck_pct:.2f}%")

        with kpi4:
            dept_count = data['department'].nunique()
            loc_count = data['departure_loc'].nunique()
            st.metric(label="Departments / Locations",
                      value=f"{dept_count} / {loc_count}")

        # Show bottleneck analysis
        st.write("### Bottleneck Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Bottleneck distribution by department
            bottleneck_dept = pd.crosstab(
                data['department'], data['bottleneck_flag'],
                normalize='index'
            ).reset_index()

            bottleneck_dept = bottleneck_dept.rename(
                columns={0: 'Normal', 1: 'Bottleneck'})

            bottleneck_dept_melted = pd.melt(
                bottleneck_dept,
                id_vars=['department'],
                var_name='bottleneck_status',
                value_name='percentage'
            )
            bottleneck_dept_melted['percentage'] = bottleneck_dept_melted['percentage'] * 100

            fig = px.bar(
                bottleneck_dept_melted[bottleneck_dept_melted['bottleneck_status']
                                       == 'Bottleneck'],
                x='department',
                y='percentage',
                color='department',
                title='Bottleneck Percentage by Department'
            )
            fig.update_layout(yaxis_title='Bottleneck Percentage (%)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Resource usage comparison between bottleneck and non-bottleneck
            bottleneck_resource = data.groupby('bottleneck_flag')[
                'resource_usage'].mean().reset_index()

            bottleneck_resource['bottleneck_flag'] = bottleneck_resource['bottleneck_flag'].map({
                0: 'Normal', 1: 'Bottleneck'
            })

            fig = px.bar(
                bottleneck_resource,
                x='bottleneck_flag',
                y='resource_usage',
                color='bottleneck_flag',
                title='Average Resource Usage by Bottleneck Status'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Resource optimization recommendations
        st.write("### Resource Optimization Recommendations")

        # Calculate high resource usage departments
        dept_resource = data.groupby('department')[
            'resource_usage'].mean().sort_values(ascending=False)
        high_resource_depts = dept_resource.head(3).index.tolist()

        recommendations = [
            f"**{high_resource_depts[0]}** department has the highest average resource usage. Consider resource redistribution.",
            f"Departments with bottlenecks should optimize utility allocation. Focus on **{bottleneck_dept_melted[bottleneck_dept_melted['bottleneck_status'] == 'Bottleneck'].sort_values('percentage', ascending=False).iloc[0]['department']}**.",
            "Implement just-in-time inventory management to reduce overstocking.",
            "Consider dynamic resource allocation based on forecasted demand patterns.",
            "Develop contingency plans for high-resource departments during peak times."
        ]

        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")

        # Show seasonal patterns
        st.write("### Seasonal Patterns")

        data_with_date = data.copy()
        data_with_date['time'] = pd.to_datetime(data_with_date['time'])
        data_with_date['month'] = data_with_date['time'].dt.month

        monthly_resource = data_with_date.groupby(
            'month')['resource_usage'].mean().reset_index()

        monthly_utility = data_with_date.groupby(
            'month')['utility_allocation'].mean().reset_index()

        # Map month numbers to month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        monthly_resource['month_name'] = monthly_resource['month'].map(
            month_names)
        monthly_utility['month_name'] = monthly_utility['month'].map(
            month_names)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_resource['month_name'],
            y=monthly_resource['resource_usage'],
            mode='lines+markers',
            name='Resource Usage'
        ))

        fig.add_trace(go.Scatter(
            x=monthly_utility['month_name'],
            y=monthly_utility['utility_allocation'],
            mode='lines+markers',
            name='Utility Allocation'
        ))

        fig.update_layout(
            title='Monthly Patterns in Resource Usage and Utility Allocation',
            xaxis_title='Month',
            yaxis_title='Value'
        )
        st.plotly_chart(fig, use_container_width=True)


# Run the app
if __name__ == "__main__":
    main()
