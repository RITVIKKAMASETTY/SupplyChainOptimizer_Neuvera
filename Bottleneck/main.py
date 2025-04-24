import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Page configuration and styling
st.set_page_config(
    page_title="Supply Chain Bottleneck Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with additional logo styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .safe {
        color: green;
        font-weight: bold;
    }
    .bottleneck {
        color: #ff0000;
        font-weight: bold;
    }
    .dashboard-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .logo-image {
        max-width: 150px;
        border-radius: 10px;
    }
    .route-selector {
        margin-top: 1rem;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# Remove the logo code block
st.markdown('<h1 class="main-header">Supply Chain Bottleneck Detection</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Upload your supply chain data or use our sample data to predict bottlenecks!</h2>', unsafe_allow_html=True)

# Add a sidebar with a better logo
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://cdn.pixabay.com/photo/2018/09/27/09/22/artificial-intelligence-3706562_1280.jpg" width="220" style="border-radius: 10px; margin-bottom: 10px;">
        <h2>Control Panel</h2>
    </div>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Option to use sample data or upload new data
data_option = st.sidebar.radio("Choose data source:", ("Use Sample Data", "Upload My Data"))

# Create containers for better layout
header_container = st.container()
dataset_container = st.container()
model_container = st.container()
metrics_container = st.container()
charts_container = st.container()

# Load data
with dataset_container:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Dataset Information</h3>', unsafe_allow_html=True)
    
    if data_option == "Use Sample Data":
        df = pd.read_csv("data.csv")
        st.success("✅ Sample data loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        else:
            st.warning("⚠️ Please upload a CSV file to continue")
            df = None
    
    if df is not None:
        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Show data overview
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Preview:**")
            st.dataframe(df.head(), height=200)
        with col2:
            st.write("**Data Summary:**")
            st.write(f"• Total Records: {len(df)}")
            st.write(f"• Departments: {', '.join(df['department'].unique())}")
            st.write(f"• Time Range: {df['time'].min().date()} to {df['time'].max().date()}")
            st.write(f"• Bottleneck Rate: {(df['status'] == 'Bottleneck').mean()*100:.1f}%")
            
    st.markdown('</div>', unsafe_allow_html=True)

# Process data if available
if 'df' in locals() and df is not None:
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check for and handle missing values
    if df_processed.isnull().any().any():
        st.warning("⚠️ Missing values detected in the dataset. Filling with appropriate values.")
        # Fill missing numerical values with median
        for col in ['resource_usage', 'utility_allocation', 'shelf_life']:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Fill missing categorical values with mode
        for col in ['priority', 'return_status', 'festive_season', 'status']:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Ensure all departments are standardized (handling new departments in validation data)
    standard_depts = ['Logistics', 'Production', 'Distribution', 'Inventory']
    if not all(dept in standard_depts for dept in df_processed['department'].unique()):
        st.info("ℹ️ Non-standard departments detected. These will be handled in the analysis.")
        # Map non-standard departments to standard ones for model compatibility
        dept_mapping = {
            'Sales': 'Distribution',
            'Marketing': 'Distribution',
            'Operations': 'Logistics',
            'Research': 'Production',
            'Finance': 'Inventory'
        }
        # Create department_mapped for model but keep original department for display
        df_processed['department_mapped'] = df_processed['department'].map(
            lambda x: dept_mapping.get(x, x)
        )
    else:
        df_processed['department_mapped'] = df_processed['department']
    
    # Prepare training data
    X_train = df_processed[['resource_usage', 'utility_allocation', 'priority', 'return_status', 'festive_season', 'shelf_life']]
    
    # Handle categorical variables properly
    priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    binary_mapping = {'No': 0, 'Yes': 1}
    status_mapping = {'Safe': 0, 'Bottleneck': 1}
    
    # Apply mappings with error handling
    X_train['priority'] = X_train['priority'].map(priority_mapping).fillna(1)  # Default to Medium if unknown
    X_train['return_status'] = X_train['return_status'].map(binary_mapping).fillna(0)  # Default to No if unknown
    X_train['festive_season'] = X_train['festive_season'].map(binary_mapping).fillna(0)  # Default to No if unknown
    
    # Convert to numeric to ensure no string values remain
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    
    # Fill any remaining NaNs from coercion
    X_train = X_train.fillna(X_train.median())
    
    # Handle target variable
    if 'status' not in df_processed.columns:
        st.error("⚠️ Status column missing. Unable to train model.")
    else:
        y_train = df_processed['status'].map(status_mapping)
        # Handle any NaN in y_train (if status had unknown values)
        if y_train.isnull().any():
            st.warning("⚠️ Unknown status values detected. These will be treated as 'Safe'.")
            y_train = y_train.fillna(0)  # Default unknown statuses to 'Safe'
        
        # Mapping for display
        df_processed['priority_score'] = df_processed['priority'].map(priority_mapping).fillna(1)
        df_processed['return_status_binary'] = df_processed['return_status'].map(binary_mapping).fillna(0)
        df_processed['festive_season_binary'] = df_processed['festive_season'].map(binary_mapping).fillna(0)
        df_processed['bottleneck_binary'] = df_processed['status'].map(status_mapping).fillna(0)
        
        # Calculate route risk score
        df_processed['risk_score'] = (
            df_processed['resource_usage'] * 0.4 + 
            (100 - df_processed['utility_allocation']) * 0.3 + 
            df_processed['priority_score'] * 10 * 0.1 + 
            df_processed['return_status_binary'] * 15 + 
            df_processed['festive_season_binary'] * 15 -
            df_processed['shelf_life'] * 0.2
        )
        
        # Check for any extreme values in risk score
        if (df_processed['risk_score'] > 200).any() or (df_processed['risk_score'] < -100).any():
            st.warning("⚠️ Extreme risk scores detected. This might indicate unusual data patterns.")
            # Clip risk scores to reasonable range
            df_processed['risk_score'] = df_processed['risk_score'].clip(-100, 200)

        # Train and save model
        with model_container:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Model Training & Prediction</h3>', unsafe_allow_html=True)
            
            try:
                model_col1, model_col2 = st.columns(2)
                
                with model_col1:
                    # Create model with exception handling
                    try:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        joblib.dump(model, 'rf_model_v2.joblib')
                        
                        # Show feature importance
                        feature_importance = pd.DataFrame({
                            'Feature': X_train.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='Feature Importance',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        model_trained = True
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        st.info("Using pre-trained model if available, or showing limited functionality.")
                        model_trained = False
                
                with model_col2:
                    # Load model for prediction with error handling
                    try:
                        if 'model_trained' not in locals() or not model_trained:
                            # Try to load existing model if training failed
                            try:
                                model = joblib.load('rf_model_v2.joblib')
                                st.info("Using previously trained model for predictions.")
                            except:
                                st.warning("No model available. Showing limited analysis.")
                                # Create a basic model with defaults just for interface
                                model = RandomForestClassifier(n_estimators=10)
                                if len(X_train) > 0 and len(y_train) > 0:
                                    # Train on at least a sample if possible
                                    sample_size = min(len(X_train), 10)
                                    model.fit(X_train.head(sample_size), y_train.head(sample_size))
                                else:
                                    # Create dummy data for interface
                                    dummy_X = pd.DataFrame({
                                        'resource_usage': [80, 90],
                                        'utility_allocation': [60, 50],
                                        'priority': [1, 2],
                                        'return_status': [0, 1],
                                        'festive_season': [0, 1],
                                        'shelf_life': [30, 20]
                                    })
                                    dummy_y = pd.Series([0, 1])
                                    model.fit(dummy_X, dummy_y)
                                    st.warning("Using demo model - predictions not reliable.")
                        
                        # Prepare latest data point for prediction with error handling
                        try:
                            if len(X_train) > 0:
                                # Find the most critical route (highest risk score) instead of just the last one
                                if 'risk_score' in df_processed.columns and len(df_processed) > 0:
                                    # Get the index of the route with highest risk score
                                    critical_idx = df_processed['risk_score'].idxmax()
                                    # Get the corresponding row from X_train
                                    # Map the dataframe index to X_train index (they might be different)
                                    if critical_idx < len(X_train):
                                        X_new = X_train.iloc[[critical_idx]].copy()
                                    else:
                                        # Fallback to the last row if index mapping issue
                                        X_new = X_train.iloc[-1:].copy()
                                        st.info("Using last route as fallback. Index mapping issue detected.")
                                else:
                                    X_new = X_train.iloc[-1:].copy()  # Fallback to last row if no risk score column
                            else:
                                # Create dummy data if no sample available
                                X_new = pd.DataFrame({
                                    'resource_usage': [85],
                                    'utility_allocation': [55],
                                    'priority': [1],
                                    'return_status': [0],
                                    'festive_season': [0],
                                    'shelf_life': [25]
                                })
                                st.warning("Using sample data for prediction demo.")
                            
                            prediction = model.predict(X_new)
                            probability = model.predict_proba(X_new)[:, 1]
                            kpi = 100 - (probability[0] * 100)
                            
                            st.write("### Prediction Details")
                            
                            # Get the critical route information
                            if len(df_processed) > 0:
                                if 'risk_score' in df_processed.columns:
                                    # Get the most critical route
                                    critical_route = df_processed.loc[df_processed['risk_score'].idxmax()]
                                    st.markdown("**🚨 Showing prediction for the most critical route (highest risk score)**")
                                else:
                                    critical_route = df_processed.iloc[-1]  # Fallback to last row
                                
                                st.write(f"**Route:** {critical_route['departure_loc']} → {critical_route['arrival_loc']}")
                                st.write(f"**Department:** {critical_route['department']}")
                                st.write(f"**Resource Usage:** {critical_route['resource_usage']}%")
                                st.write(f"**Utility Allocation:** {critical_route['utility_allocation']}%") 
                                st.write(f"**Priority:** {critical_route['priority']}")
                                if 'risk_score' in df_processed.columns:
                                    st.write(f"**Risk Score:** {critical_route['risk_score']:.1f}")
                                
                                # Enhanced verdict function with error handling
                                def get_verdict(prediction, kpi, row):
                                    try:
                                        if prediction[0] == 1:
                                            festive_text = "festive demand" if row.get('festive_season') == 'Yes' else ""
                                            returns_text = "returns processing" if row.get('return_status') == 'Yes' else ""
                                            conjunction = " and " if festive_text and returns_text else ""
                                            factors = festive_text + conjunction + returns_text
                                            factors_text = f" and {factors}" if factors else ""
                                            
                                            return (f"<div class='bottleneck'>⚠️ <b>Bottleneck Alert!</b></div> A bottleneck is detected in {row['department']} from {row['departure_loc']} to {row['arrival_loc']} due to high resource use ({row['resource_usage']}%), low utility ({row['utility_allocation']}%){factors_text}. Shelf life: {row['shelf_life']} days.",
                                                    [f"Add resources at {row['departure_loc']}", 
                                                     f"Adjust utility by {row.get('deadline', 'next week')}", 
                                                     f"Prioritize shipments with shorter shelf life", 
                                                     f"Review return policy during peak seasons"], kpi)
                                        else:
                                            return (f"<div class='safe'>✅ <b>All Good!</b></div> The {row['department']} route from {row['departure_loc']} to {row['arrival_loc']} is safe with {row['resource_usage']}% usage. Shelf life: {row['shelf_life']} days.",
                                                    ["Maintain current plan", 
                                                     f"Review by {row.get('deadline', 'next month')}", 
                                                     "Consider optimizing resource allocation"], kpi)
                                    except Exception as e:
                                        # Fallback verdict if there's an error
                                        is_bottleneck = prediction[0] == 1
                                        status = "Bottleneck Alert" if is_bottleneck else "Route is Safe"
                                        color_class = "bottleneck" if is_bottleneck else "safe"
                                        icon = "⚠️" if is_bottleneck else "✅"
                                        
                                        return (f"<div class='{color_class}'>{icon} <b>{status}!</b></div> Analysis prediction based on input data.",
                                                ["Review resource allocation", 
                                                 "Analyze supply chain patterns", 
                                                 "Monitor high-risk routes"], kpi)
                                
                                verdict_text, steps, kpi_value = get_verdict(prediction, kpi, critical_route)
                            else:
                                # Fallback if no route data
                                status = "Bottleneck Alert" if prediction[0] == 1 else "Route is Safe"
                                color_class = "bottleneck" if prediction[0] == 1 else "safe"
                                icon = "⚠️" if prediction[0] == 1 else "✅"
                                kpi_value = kpi
                                
                                verdict_text = f"<div class='{color_class}'>{icon} <b>{status}!</b></div> Sample prediction (no route data available)."
                                steps = ["Review resource allocation", "Analyze supply chain patterns", "Monitor high-risk routes"]
                                
                                st.write("**Sample Prediction (No Route Data)**")
                                st.write("This is a demonstration using sample data.")
                        except Exception as e:
                            st.error(f"Error in prediction: {str(e)}")
                            # Create fallback values for the rest of the interface
                            prediction = [0]
                            kpi_value = 75.0
                            verdict_text = "<div class='safe'>✅ <b>Sample Analysis</b></div> This is a placeholder due to prediction errors."
                            steps = ["Check data quality", "Ensure all required columns are present", "Review input values for anomalies"]
                    except Exception as e:
                        st.error(f"Error in model loading or prediction: {str(e)}")
                        # Create fallback values for the rest of the interface
                        kpi_value = 50.0
                        verdict_text = "<div class='safe'>⚠️ <b>Analysis Limited</b></div> Error in model processing."
                        steps = ["Check data quality", "Ensure all required columns are present", "Try a different dataset"]
                
                st.markdown(verdict_text, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in model processing: {str(e)}")
                st.info("Try using a different dataset or check the data format.")
                # Create fallback values for the rest of the interface
                kpi_value = 50.0
                verdict_text = "Analysis Limited"
                steps = ["Check data quality", "Ensure all required columns are present", "Try a different dataset"]
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Key metrics and top vulnerable routes
        with metrics_container:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Key Metrics & Vulnerable Routes</h3>', unsafe_allow_html=True)
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.write("### Safety Score")
                
                # Create a gauge chart for safety score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=kpi_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Safety Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "red"},
                            {'range': [40, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': kpi_value
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("### Quick Tips")
                for i, step in enumerate(steps):
                    st.write(f"**{i+1}.** {step}")
            
            with metrics_col2:
                st.write("### Top 5 Most Vulnerable Routes")
                
                # Calculate risk for all routes and find the most vulnerable ones
                top_vulnerable = df_processed.sort_values('risk_score', ascending=False).head(5)
                
                # Create a table for the vulnerable routes
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=['Departure', 'Arrival', 'Department', 'Risk Score', 'Status'],
                        fill_color='royalblue',
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=[
                            top_vulnerable['departure_loc'],
                            top_vulnerable['arrival_loc'],
                            top_vulnerable['department'],
                            top_vulnerable['risk_score'].round(1),
                            top_vulnerable['status']
                        ],
                        fill_color=[['white', 'lightgrey']*5],
                        align='left',
                        font_color=[['black']*5, ['red' if x == 'Bottleneck' else 'green' for x in top_vulnerable['status']]]
                    )
                )])
                
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add dropdown to select any of the top 5 routes for analysis
                st.markdown('<div class="route-selector">', unsafe_allow_html=True)
                st.write("### Analyze Critical Route")
                
                # Create options for dropdown
                route_options = [f"{row['departure_loc']} → {row['arrival_loc']} (Risk: {row['risk_score']:.1f})" 
                                for _, row in top_vulnerable.iterrows()]
                default_idx = 0  # Default to the most critical route
                
                selected_route = st.selectbox(
                    "Select a route to analyze:",
                    options=route_options,
                    index=default_idx
                )
                
                # Find the selected route index in the top_vulnerable dataframe
                selected_idx = route_options.index(selected_route)
                selected_route_data = top_vulnerable.iloc[selected_idx]
                
                if st.button("Analyze Selected Route", use_container_width=True):
                    st.markdown("#### Route Analysis")
                    
                    # Show detailed information about the selected route
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Route:** {selected_route_data['departure_loc']} → {selected_route_data['arrival_loc']}")
                        st.markdown(f"**Department:** {selected_route_data['department']}")
                        st.markdown(f"**Status:** {'🔴 Bottleneck' if selected_route_data['status'] == 'Bottleneck' else '🟢 Safe'}")
                        st.markdown(f"**Risk Score:** {selected_route_data['risk_score']:.1f}")
                    
                    with col2:
                        st.markdown(f"**Resource Usage:** {selected_route_data['resource_usage']}%")
                        st.markdown(f"**Utility Allocation:** {selected_route_data['utility_allocation']}%")
                        st.markdown(f"**Priority:** {selected_route_data['priority']}")
                        st.markdown(f"**Shelf Life:** {selected_route_data['shelf_life']} days")
                    
                    # Calculate and show risk factors
                    st.markdown("#### Risk Factors")
                    
                    # Base factors to assess
                    risk_factors = []
                    
                    # Resource usage assessment
                    if selected_route_data['resource_usage'] > 90:
                        risk_factors.append(("Extreme Resource Usage", "Resource utilization is critically high (>90%). This is a primary bottleneck factor."))
                    elif selected_route_data['resource_usage'] > 80:
                        risk_factors.append(("High Resource Usage", "Resource utilization is high (>80%). Consider allocating additional resources."))
                    
                    # Utility allocation assessment
                    if selected_route_data['utility_allocation'] < 40:
                        risk_factors.append(("Low Utility Allocation", "Utility allocation is very low (<40%). Efficiency is significantly compromised."))
                    elif selected_route_data['utility_allocation'] < 60:
                        risk_factors.append(("Moderate Utility Allocation", "Utility allocation is below optimal levels (<60%). Review allocation strategy."))
                    
                    # Priority assessment
                    if selected_route_data['priority'] == 'High':
                        risk_factors.append(("High Priority Shipment", "This is a high priority route which increases its criticality in the supply chain."))
                    
                    # Festive season impact
                    if selected_route_data['festive_season'] == 'Yes':
                        risk_factors.append(("Festive Season Impact", "This route is operating during festive season, which increases demand pressure."))
                    
                    # Return status impact
                    if selected_route_data['return_status'] == 'Yes':
                        risk_factors.append(("Return Processing Required", "This route requires return processing, adding complexity to logistics."))
                    
                    # Shelf life assessment
                    if selected_route_data['shelf_life'] < 20:
                        risk_factors.append(("Short Shelf Life", f"Products have a short shelf life ({selected_route_data['shelf_life']} days), increasing time pressure."))
                    
                    # Display the risk factors
                    if risk_factors:
                        for factor, description in risk_factors:
                            st.markdown(f"**{factor}:** {description}")
                    else:
                        st.markdown("No significant risk factors identified for this route.")
                    
                    # Provide recommendations based on the analysis
                    st.markdown("#### Recommendations")
                    
                    if selected_route_data['status'] == 'Bottleneck':
                        recommendations = [
                            f"Increase resource allocation at {selected_route_data['departure_loc']} by at least {max(5, int(95 - selected_route_data['resource_usage']))}%",
                            f"Improve utility allocation to at least {min(80, int(selected_route_data['utility_allocation'] + 20))}%",
                            "Implement priority-based scheduling for this route",
                            "Consider alternative routes to reduce pressure"
                        ]
                        
                        # Add specific recommendations based on risk factors
                        if selected_route_data['festive_season'] == 'Yes':
                            recommendations.append("Prepare for festive season demand with additional temporary resources")
                        
                        if selected_route_data['return_status'] == 'Yes':
                            recommendations.append("Set up a dedicated return processing team")
                        
                        if selected_route_data['shelf_life'] < 20:
                            recommendations.append("Prioritize expedited shipping for these short shelf-life products")
                    else:
                        recommendations = [
                            "Maintain current resource levels",
                            "Monitor for seasonal variations",
                            "Review performance weekly",
                            "Document effective practices from this route"
                        ]
                    
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"**{i+1}.** {rec}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced visualizations
        with charts_container:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Advanced Analytics</h3>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Resource Analysis", "Bottleneck Patterns", "Department Performance"])
            
            with tab1:
                try:
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Resource usage by department
                        fig = px.box(
                            df_processed, 
                            x='department', 
                            y='resource_usage',
                            color='status',
                            title='Resource Usage by Department',
                            color_discrete_map={'Safe': 'green', 'Bottleneck': 'red'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        # Resource vs Utility scatter plot
                        fig = px.scatter(
                            df_processed, 
                            x='resource_usage', 
                            y='utility_allocation',
                            color='status',
                            size='shelf_life',
                            hover_name='departure_loc',
                            hover_data=['arrival_loc', 'department', 'priority'],
                            title='Resource Usage vs Utility Allocation',
                            color_discrete_map={'Safe': 'green', 'Bottleneck': 'red'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating resource analysis charts: {str(e)}")
                    st.info("This can happen with unusual data. Try with a different dataset.")
            
            with tab2:
                try:
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Bottleneck by department and festive season
                        # Use department_mapped instead of department if available for consistency
                        dept_col = 'department_mapped' if 'department_mapped' in df_processed.columns else 'department'
                        
                        # Create cross-tabulation with error handling
                        try:
                            pivot_data = pd.crosstab(
                                df_processed[dept_col], 
                                df_processed['festive_season'],
                                values=df_processed['bottleneck_binary'],
                                aggfunc='mean'
                            ).fillna(0) * 100
                            
                            fig = px.imshow(
                                pivot_data,
                                text_auto='.1f',
                                labels=dict(x='Festive Season', y='Department', color='Bottleneck %'),
                                title='Bottleneck % by Department & Festive Season',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            # Fallback to simpler visualization if cross-tab fails
                            fallback_df = df_processed.groupby(dept_col)['bottleneck_binary'].mean().reset_index()
                            fallback_df['bottleneck_pct'] = fallback_df['bottleneck_binary'] * 100
                            
                            fig = px.bar(
                                fallback_df,
                                x=dept_col,
                                y='bottleneck_pct',
                                title='Bottleneck % by Department',
                                color='bottleneck_pct',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("Simplified visualization shown due to data constraints.")
                    
                    with chart_col2:
                        # Bottleneck trend over time
                        if len(df_processed['time'].dt.date.unique()) > 1:
                            time_trend = df_processed.groupby(df_processed['time'].dt.date)['bottleneck_binary'].mean() * 100
                            
                            fig = px.line(
                                x=time_trend.index, 
                                y=time_trend.values,
                                markers=True,
                                labels={'x': 'Date', 'y': 'Bottleneck %'},
                                title='Bottleneck Trend Over Time'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough date range for time trend analysis.")
                            # Alternative visualization for limited time data
                            alt_fig = px.pie(
                                df_processed, 
                                names='status',
                                title='Status Distribution',
                                color='status',
                                color_discrete_map={'Safe': 'green', 'Bottleneck': 'red'}
                            )
                            alt_fig.update_layout(height=400)
                            st.plotly_chart(alt_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating bottleneck pattern charts: {str(e)}")
                    st.info("This can happen with unusual data. Try with a different dataset.")
            
            with tab3:
                try:
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Department performance
                        # Use department_mapped for grouping if available
                        dept_col = 'department_mapped' if 'department_mapped' in df_processed.columns else 'department'
                        
                        # Only include columns that exist for all department types
                        agg_cols = {
                            'resource_usage': 'mean',
                            'utility_allocation': 'mean',
                            'bottleneck_binary': 'mean'
                        }
                        
                        if 'risk_score' in df_processed.columns:
                            agg_cols['risk_score'] = 'mean'
                        
                        dept_perf = df_processed.groupby(dept_col).agg(agg_cols)
                        
                        # Choose the most appropriate metric for sorting
                        if 'risk_score' in dept_perf.columns:
                            sort_col = 'risk_score'
                        else:
                            sort_col = 'bottleneck_binary'
                            
                        dept_perf = dept_perf.sort_values(sort_col, ascending=False)
                        
                        # Check if there's enough data to plot
                        if len(dept_perf) > 0:
                            fig = px.bar(
                                dept_perf.reset_index(), 
                                x=dept_col, 
                                y=sort_col,
                                color='bottleneck_binary',
                                title='Department Risk Assessment',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Not enough department data to create visualization.")
                    
                    with chart_col2:
                        # Resource usage trend or alternative view
                        if len(df_processed['time'].dt.date.unique()) > 1 and len(df_processed['department'].unique()) <= 8:
                            # Resource usage trend if we have enough time points and not too many departments
                            fig = px.line(
                                df_processed, 
                                x='time', 
                                y='resource_usage',
                                color='department',
                                line_dash='status',
                                title='Resource Usage Trend by Department',
                                markers=True
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Alternative view: department vs resource usage and utility allocation
                            radar_data = df_processed.groupby('department').agg({
                                'resource_usage': 'mean',
                                'utility_allocation': 'mean',
                                'shelf_life': 'mean',
                                'risk_score': 'mean'
                            }).reset_index()
                            
                            # If too many departments, show top 6
                            if len(radar_data) > 6:
                                radar_data = radar_data.sort_values('risk_score', ascending=False).head(6)
                            
                            fig = px.scatter_polar(
                                radar_data,
                                r='resource_usage',
                                theta='department',
                                size='risk_score',
                                color='utility_allocation',
                                title='Department Resource Profile',
                                color_continuous_scale='Bluered'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("Showing department profile view due to data constraints.")
                except Exception as e:
                    st.error(f"Error generating department performance charts: {str(e)}")
                    st.info("This can happen with unusual data. Try with a different dataset.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Save results
        if st.sidebar.button("Save Results", use_container_width=True):
            try:
                with open("results.txt", "w") as f:
                    # Clean verdict text for saving (remove HTML tags)
                    clean_verdict = re.sub('<.*?>', '', verdict_text)
                    
                    # Include the critical route information in the saved results
                    if 'risk_score' in df_processed.columns:
                        critical_route = df_processed.loc[df_processed['risk_score'].idxmax()]
                        f.write(f"BOTTLENECK DETECTION REPORT\n{'='*30}\n\n")
                        f.write(f"CRITICAL ROUTE ANALYSIS:\n")
                        f.write(f"Route: {critical_route['departure_loc']} → {critical_route['arrival_loc']}\n")
                        f.write(f"Department: {critical_route['department']}\n")
                        f.write(f"Risk Score: {critical_route['risk_score']:.1f}\n\n")
                    
                    f.write(f"Verdict: {clean_verdict}\nSafety Score: {kpi_value:.1f}%\nTips: {', '.join(steps)}")
                    f.write("\n\nTop 5 Vulnerable Routes:\n")
                    
                    for i, row in top_vulnerable.iterrows():
                        f.write(f"{i+1}. {row['departure_loc']} → {row['arrival_loc']} ({row['department']}): Risk Score {row['risk_score']:.1f}\n")
                    
                    # Add analysis timestamp
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\nAnalysis generated on: {timestamp}")
                
                st.sidebar.success("Results saved as 'results.txt'!")
            except Exception as e:
                st.sidebar.error(f"Error saving results: {str(e)}")
            
        # Download option
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Download Options")
        
        # Create a download function for CSV
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_data = convert_df_to_csv(df_processed)
        st.sidebar.download_button(
            "Download Processed Data",
            csv_data,
            "bottleneck_analysis.csv",
            "text/csv",
            key="download-csv",
            use_container_width=True
        )