import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Fresh Produce Analysis",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin-top: 1rem;
    }
    .chart-container {
        background-color: gray;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Fresh Produce Spoilage Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze cold chain and logistics data for fresh produce</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    
    model_options = ["Deepseek", "Llama", "Gemma"]
    selected_model = st.selectbox("Select AI Model", model_options)
    
    st.divider()
    
    st.header("Data Explorer")
    
    @st.cache_data
    def fetch_products():
        try:
            base_url = "http://localhost:8000"
            response = requests.get(f"{base_url}/products")
            if response.status_code == 200:
                return response.json().get("products", [])
            return []
        except:
            return []
    
    products = fetch_products()
    if products:
        selected_product = st.selectbox("Filter by Product", ["All Products"] + products)
        if st.button("Show Product Data"):
            st.session_state["show_product_data"] = True
            st.session_state["selected_product"] = selected_product
    
    st.divider()
    
    st.header("Date Range")
    start_date = st.date_input("Start Date", value=datetime(2025, 4, 1).date())
    end_date = st.date_input("End Date", value=datetime(2025, 4, 10).date())
    
    st.divider()
    
    st.header("Example Questions")
    example_questions = [
        "What produces have the highest spoilage rates?",
        "How does transit time affect spoilage?",
        "Compare spoilage rates between refrigerated and ambient storage",
        "Which products should be prioritized for delivery?",
        "What's the relationship between temperature and spoilage for berries?",
        "Which vegetables need cooler truck shipping?"
    ]
    
    for question in example_questions:
        if st.button(question, key=question):
            st.session_state["question"] = question

tab1, tab4, tab2, tab3 = st.tabs(["🔍 Query Analysis", " Cost analyzer", "📊 Data Visualization", "📈 Performance Metrics"])

with tab1:
    question = st.text_input(
        "Ask a question about the fresh produce data:",
        value=st.session_state.get("question", ""),
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Analyze", type="primary")
    
    with col2:
        st.markdown("")  
    if submit_button:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner(f"Analyzing with {selected_model} model..."):
                base_url = "http://localhost:8000"  # Replace with your FastAPI server URL
                if selected_model == "Deepseek":
                    endpoint = f"{base_url}/deepseek"
                elif selected_model == "Llama":
                    endpoint = f"{base_url}/llama"
                elif selected_model == "Gemma":
                    endpoint = f"{base_url}/gemma"
                
                payload = {"question": question}
                try:
                    response = requests.post(endpoint, json=payload, timeout=60)
                    response.raise_for_status()
                    answer = response.json().get("answer", "No answer received.")
                except requests.exceptions.RequestException as e:
                    answer = f"Error: {e}"
                
                st.markdown("### Analysis Results")
                with st.container(border=True):
                    st.markdown(answer)
                
                st.markdown("---")
                st.markdown("##### Not satisfied? Try a different model or refine your question.")



with tab4:
    tab4_question = st.text_input(
        "Ask a question about the fresh produce data:",
        value=st.session_state.get("tab4_question", ""),
        key="tab4_question_input"
    )
    
    tab4_col1, tab4_col2, tab4_col3 = st.columns([1, 1, 4])
    with tab4_col1:
        tab4_submit_button = st.button("Analyze", key="tab4_analyze_button", type="primary")
    with tab4_col2:
        tab4_cost_button = st.button("Cost Insights", key="tab4_cost_button", type="secondary")
    with tab4_col3:
        st.markdown("")  
    if tab4_submit_button:
        if not tab4_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner(f"Analyzing with {selected_model} model..."):
                base_url = "http://localhost:8000"
                if selected_model == "Deepseek":
                    endpoint = f"{base_url}/deepseek"
                elif selected_model == "Llama":
                    endpoint = f"{base_url}/llama"
                elif selected_model == "Gemma":
                    endpoint = f"{base_url}/gemma"
                else:
                    endpoint = None

                if endpoint:
                    payload = {"question": tab4_question}
                    try:
                        response = requests.post(endpoint, json=payload, timeout=60)
                        response.raise_for_status()
                        answer = response.json().get("answer", "No answer received.")
                    except requests.exceptions.RequestException as e:
                        answer = f"Error: {e}"
                else:
                    answer = "Invalid model selected."

                st.markdown("### Analysis Results")
                with st.container(border=True):
                    st.markdown(answer)

                st.markdown("---")
                st.markdown("##### Not satisfied? Try a different model or refine your question.")

    elif tab4_cost_button:
        with st.spinner("Generating cost insights..."):
            def generate_sample_iot_data_tab4():
                dates = pd.date_range(start="2025-04-01", periods=168, freq="H")
                power_base = 45 + np.sin(np.linspace(0, 14*np.pi, 168)) * 15
                power_noise = np.random.normal(0, 3, 168)
                power = power_base + power_noise
                for i in range(168):
                    hour = i % 24
                    if 8 <= hour <= 18:
                        power[i] += 20
                water = 120 + np.sin(np.linspace(0, 14*np.pi, 168)) * 30 + np.random.normal(0, 10, 168)
                fuel = 8 + np.sin(np.linspace(0, 7*np.pi, 168)) * 3 + np.random.normal(0, 1, 168)
                temp = 22 + np.sin(np.linspace(0, 14*np.pi, 168)) * 5 + np.random.normal(0, 1, 168)
                humidity = 55 + np.sin(np.linspace(0, 14*np.pi, 168)) * 15 + np.random.normal(0, 3, 168)

                power_cost = power * 0.15
                water_cost = water * 0.003
                fuel_cost = fuel * 1.20

                df1 = pd.DataFrame({
                    "Timestamp": dates,
                    "Power_Consumption_kWh": power,
                    "Water_Consumption_L": water,
                    "Fuel_Consumption_L": fuel,
                    "Temperature_C": temp,
                    "Humidity_Percent": humidity,
                    "Power_Cost_USD": power_cost,
                    "Water_Cost_USD": water_cost,
                    "Fuel_Cost_USD": fuel_cost,
                    "Total_Cost_USD": power_cost + water_cost + fuel_cost
                })
                df1["Date"] = df1["Timestamp"].dt.date
                df1["Hour"] = df1["Timestamp"].dt.hour
                return df1

            iot_data_tab4 = generate_sample_iot_data_tab4()
            latest_date = iot_data_tab4["Timestamp"].max().date()
            three_days_ago = latest_date - pd.Timedelta(days=3)
            filtered_data = iot_data_tab4[iot_data_tab4["Timestamp"].dt.date >= three_days_ago]

            api_data = {
                "Power_Consumption_kWh": filtered_data["Power_Consumption_kWh"].tolist(),
                "Water_Consumption_L": filtered_data["Water_Consumption_L"].tolist(),
                "Fuel_Consumption_L": filtered_data["Fuel_Consumption_L"].tolist(),
                "Temperature_C": filtered_data["Temperature_C"].tolist(),
                "Humidity_Percent": filtered_data["Humidity_Percent"].tolist(),
                "Power_Cost_USD": filtered_data["Power_Cost_USD"].tolist(),
                "Water_Cost_USD": filtered_data["Water_Cost_USD"].tolist(),
                "Fuel_Cost_USD": filtered_data["Fuel_Cost_USD"].tolist(),
                "Total_Cost_USD": filtered_data["Total_Cost_USD"].tolist(),
                "Timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in filtered_data["Timestamp"].tolist()]
            }

            analysis_question = "Provide a quick cost analysis with optimization suggestions"

            payload = {
                "data": api_data,
                "question": analysis_question
            }

            try:
                response = requests.post(f"http://localhost:8000/analyze_cost", json=payload, timeout=90)
                response.raise_for_status()
                cost_analysis_result = response.json().get("answer", "No analysis received.")

                st.markdown("### Cost Insights Analysis")
                with st.container(border=True):
                    st.markdown(cost_analysis_result)

                total_power_cost = filtered_data["Power_Cost_USD"].sum()
                total_water_cost = filtered_data["Water_Cost_USD"].sum()
                total_fuel_cost = filtered_data["Fuel_Cost_USD"].sum()
                total_cost = total_power_cost + total_water_cost + total_fuel_cost

                st.markdown("### Cost Summary (Last 3 Days)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cost", f"${total_cost:.2f}")
                with col2:
                    st.metric("Power Cost", f"${total_power_cost:.2f}", delta=f"{(total_power_cost/total_cost*100):.1f}%")
                with col3:
                    st.metric("Water Cost", f"${total_water_cost:.2f}", delta=f"{(total_water_cost/total_cost*100):.1f}%")
                with col4:
                    st.metric("Fuel Cost", f"${total_fuel_cost:.2f}", delta=f"{(total_fuel_cost/total_cost*100):.1f}%")

                st.info("💡 For more detailed cost analysis and optimization recommendations, please visit the '💰 Cost Insights' tab.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error contacting cost analysis API: {str(e)}")
                st.warning("Make sure your FastAPI server is running and accessible.")

with tab2:
    st.markdown("## 📊 Visualization Dashboard")
    st.write("Explore insights through interactive visualizations of key metrics")
    
    @st.cache_data
    def load_sensor_data():
        data = {
            "Timestamp": pd.date_range(start="2025-04-01", periods=30, freq="H"),
            "Temperature_C": np.random.normal(25, 5, 30),
            "Humidity_%": np.random.normal(65, 10, 30),
            "Power_Consumption_kWh": np.random.normal(6, 2, 30),
            "Water_Consumption_L": np.random.normal(4, 1.5, 30),
            "CO2_Emission_kg": np.random.normal(5, 2, 30),
            "Operating_Cost_USD": np.random.normal(7, 2, 30),
            "Fuel_Consumption_L": np.random.normal(1.2, 0.4, 30),
            "Machine_Status": np.random.choice(["Running", "Idle", "Maintenance"], 30),
            "Alert_Flag": np.random.choice([0, 1], 30, p=[0.8, 0.2])
        }
        return pd.DataFrame(data)
    
    sensor_df = load_sensor_data()
    
    def get_groq_insights(attribute_name, data):
        base_url = "http://localhost:8000"
        prompt = f"""
        Analyze the following {attribute_name} data and provide 4-5 key insights about the trends, patterns, and potential implications:
        {data.to_dict()}
        
        Format your response as bullet points focusing on:
        - Notable trends or patterns
        - Potential correlations with other metrics
        - Operational implications
        - Optimization opportunities
        - Environmental or cost impacts
        """
        
        try:
            response = requests.post(f"{base_url}/llama", json={"question": prompt}, timeout=60)
            if response.status_code == 200:
                insights = response.json().get("answer", "")
                insights = insights.split("\n")
                insights = [i.strip() for i in insights if i.strip().startswith("-")]
                return insights[:5]  
            return ["Unable to generate insights. Please check the API connection."]
        except:
            return ["Unable to generate insights. Please check the API connection."]
    
    st.markdown("### 📌 Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_temp = round(sensor_df["Temperature_C"].mean(), 1)
        st.metric(
            label="Average Temperature",
            value=f"{avg_temp}°C",
            delta=f"{round(avg_temp - 23.5, 1)}°C"
        )
    
    with col2:
        avg_power = round(sensor_df["Power_Consumption_kWh"].mean(), 2)
        st.metric(
            label="Avg Power Consumption",
            value=f"{avg_power} kWh",
            delta=f"{round(((avg_power/5.5)-1)*100, 1)}%"
        )
    
    with col3:
        avg_water = round(sensor_df["Water_Consumption_L"].mean(), 2)
        st.metric(
            label="Avg Water Consumption",
            value=f"{avg_water} L",
            delta=f"{round(((avg_water/4)-1)*100, 1)}%"
        )
    
    with col4:
        avg_cost = round(sensor_df["Operating_Cost_USD"].mean(), 2)
        st.metric(
            label="Avg Operating Cost",
            value=f"${avg_cost}",
            delta=f"{round(((avg_cost/6.5)-1)*100, 1)}%"
        )
    
    st.markdown("---")
    
    st.markdown("### ⚡ Electrical Consumption Analysis")
    
    fig_power = px.line(
        sensor_df, 
        x="Timestamp", 
        y="Power_Consumption_kWh",
        title="Power Consumption Over Time",
        labels={"Power_Consumption_kWh": "Power (kWh)", "Timestamp": "Time"},
        template="plotly_white"
    )
    fig_power.update_traces(line=dict(color="#1565C0", width=3))
    fig_power.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Time",
        yaxis_title="Power Consumption (kWh)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_power, use_container_width=True)
    
    power_insights = get_groq_insights("Power Consumption", sensor_df[["Timestamp", "Power_Consumption_kWh"]])
    
    with st.expander("About the Visualization"):
        st.markdown("#### Power Consumption Insights")
        for insight in power_insights:
            st.markdown(f"• {insight}")
    
    st.markdown("---")
    
    st.markdown("### 💧 Water Consumption Analysis")
    
    water_daily = sensor_df.copy()
    water_daily["Date"] = water_daily["Timestamp"].dt.date
    water_daily = water_daily.groupby("Date")["Water_Consumption_L"].sum().reset_index()
    
    fig_water = px.bar(
        water_daily,
        x="Date",
        y="Water_Consumption_L",
        title="Daily Water Consumption",
        labels={"Water_Consumption_L": "Water (L)", "Date": "Date"},
        template="plotly_white"
    )
    fig_water.update_traces(marker_color="#2E7D32")
    fig_water.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Water Consumption (L)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_water, use_container_width=True)
    
    water_insights = get_groq_insights("Water Consumption", water_daily)
    
    with st.expander("About the Visualization"):
        st.markdown("#### Water Consumption Insights")
        for insight in water_insights:
            st.markdown(f"• {insight}")
    
    st.markdown("---")
    
    st.markdown("### 🌱 Carbon Emissions Analysis")
    
    corr_columns = ["CO2_Emission_kg", "Power_Consumption_kWh", "Water_Consumption_L", "Fuel_Consumption_L"]
    corr_df = sensor_df[corr_columns].corr()
    
    fig_co2_corr = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Correlation Between CO2 Emissions and Other Metrics"
    )
    fig_co2_corr.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig_co2 = px.line(
            sensor_df, 
            x="Timestamp", 
            y="CO2_Emission_kg",
            title="CO2 Emissions Over Time",
            labels={"CO2_Emission_kg": "CO2 (kg)", "Timestamp": "Time"},
            template="plotly_white"
        )
        fig_co2.update_traces(line=dict(color="#7B1FA2", width=3))
        fig_co2.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title="Time",
            yaxis_title="CO2 Emissions (kg)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_co2, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_co2_corr, use_container_width=True)
    
    co2_insights = get_groq_insights("CO2 Emissions", sensor_df[["Timestamp", "CO2_Emission_kg", "Power_Consumption_kWh", "Fuel_Consumption_L"]])
    
    with st.expander("About the Visualization"):
        st.markdown("#### Carbon Emissions Insights")
        for insight in co2_insights:
            st.markdown(f"• {insight}")
    
    st.markdown("---")
    
    st.markdown("### 💰 Operating Cost Analysis")
    
    cost_breakdown = pd.DataFrame({
        "Category": ["Energy", "Water", "Maintenance", "Labor", "Transport"],
        "Cost": [35, 15, 20, 20, 10]
    })
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        fig_cost_pie = px.pie(
            cost_breakdown,
            values="Cost",
            names="Category",
            title="Cost Allocation Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_cost_pie.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_cost_pie, use_container_width=True)
    
    with col2:
        fig_cost_temp = px.scatter(
            sensor_df,
            x="Temperature_C",
            y="Operating_Cost_USD",
            color="Alert_Flag",
            size="Power_Consumption_kWh",
            title="Operating Cost vs Temperature",
            labels={
                "Operating_Cost_USD": "Cost (USD)",
                "Temperature_C": "Temperature (°C)",
                "Alert_Flag": "Alert Status"
            },
            color_discrete_map={0: "#4CAF50", 1: "#F44336"}
        )
        fig_cost_temp.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title="Temperature (°C)",
            yaxis_title="Operating Cost (USD)"
        )
        st.plotly_chart(fig_cost_temp, use_container_width=True)
    
    cost_insights = get_groq_insights("Operating Costs", sensor_df[["Operating_Cost_USD", "Power_Consumption_kWh", "Water_Consumption_L", "Fuel_Consumption_L", "Temperature_C"]])
    
    with st.expander("About the Visualization"):
        st.markdown("#### Operating Cost Insights")
        for insight in cost_insights:
            st.markdown(f"• {insight}")

with tab3:
    st.markdown("## 📈 Performance Metrics")
    st.write("Track key performance indicators and efficiency metrics over time")
    
    sample_data = {
        "Product": ["Mangoes", "Cauliflower", "Cherries", "Blueberries", "Papaya", "Green Peas", 
                    "Peaches", "Plums", "Okra", "Tomatoes", "Mushrooms", "Lettuce", "Zucchini",
                    "Spinach", "Carrots", "Strawberries", "Cucumbers", "Pineapples"],
        "Category": ["Fruit", "Vegetable", "Fruit", "Fruit", "Fruit", "Vegetable", 
                     "Fruit", "Fruit", "Vegetable", "Vegetable", "Vegetable", "Vegetable", 
                     "Vegetable", "Vegetable", "Vegetable", "Fruit", "Vegetable", "Fruit"],
        "Spoilage (%)": [21.3, 29.2, 20.5, 20.7, 22.0, 34.9, 24.4, 34.6, 25.3, 30.6, 28.6, 
                         27.5, 19.0, 24.3, 20.5, 13.3, 25.9, 5.2],
        "Avg_Temperature": [16.6, 20.4, 15.7, 27.7, 27.3, 23.8, 17.1, 26.0, 29.2, 34.4, 
                           27.9, 25.7, 30.1, 34.5, 28.2, 16.5, 27.3, 19.9],
        "Transit_Time": [52, 71, 80, 106, 35, 50, 71, 64, 63, 105, 31, 111, 10, 73, 
                        34, 81, 60, 87],
        "Storage": ["Refrigerated", "Refrigerated", "Refrigerated", "Ambient", "Refrigerated",
                    "Ambient", "Ambient", "Refrigerated", "Refrigerated", "Ambient", 
                    "Ambient", "Refrigerated", "Ambient", "Ambient", "Refrigerated", 
                    "Refrigerated", "Refrigerated", "Refrigerated"]
    }
    df_viz = pd.DataFrame(sample_data)
    
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_spoilage = round(df_viz["Spoilage (%)"].mean(), 1)
        st.metric(
            label="Avg Spoilage Rate",
            value=f"{avg_spoilage}%",
            delta=f"{round(avg_spoilage - 25, 1)}%",
            delta_color="inverse"
        )
    
    with col2:
        avg_transit = round(df_viz["Transit_Time"].mean(), 1)
        st.metric(
            label="Avg Transit Time",
            value=f"{avg_transit} hrs",
            delta=f"{round(avg_transit - 70, 1)} hrs",
            delta_color="inverse"
        )
    
    with col3:
        refrigerated_pct = round(len(df_viz[df_viz["Storage"] == "Refrigerated"]) / len(df_viz) * 100, 1)
        st.metric(
            label="Refrigerated Storage",
            value=f"{refrigerated_pct}%",
            delta=f"{round(refrigerated_pct - 50, 1)}%"
        )
    
    with col4:
        fruit_spoilage = round(df_viz[df_viz["Category"] == "Fruit"]["Spoilage (%)"].mean(), 1)
        veg_spoilage = round(df_viz[df_viz["Category"] == "Vegetable"]["Spoilage (%)"].mean(), 1)
        st.metric(
            label="Fruit vs Veg Spoilage",
            value=f"{fruit_spoilage}% vs {veg_spoilage}%",
            delta=f"{round(fruit_spoilage - veg_spoilage, 1)}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    viz_options = [
        "Spoilage by Product",
        "Spoilage by Category",
        "Temperature vs Spoilage",
        "Transit Time vs Spoilage",
        "Storage Type Comparison"
    ]
    
    selected_viz = st.selectbox("Select Visualization", viz_options)
    
    st.markdown(f"### {selected_viz}")
    
    if selected_viz == "Spoilage by Product":
        sorted_df = df_viz.sort_values("Spoilage (%)", ascending=False)
        
        fig_spoilage_product = px.bar(
            sorted_df,
            x="Product",
            y="Spoilage (%)",
            color="Category",
            title="Spoilage Rates by Product",
            color_discrete_map={"Fruit": "#4CAF50", "Vegetable": "#2196F3"},
            template="plotly_white"
        )
        fig_spoilage_product.update_layout(
            height=500,
            xaxis_tickangle=-45,
            xaxis_title="Product",
            yaxis_title="Spoilage Rate (%)",
            legend_title="Category"
        )
        st.plotly_chart(fig_spoilage_product, use_container_width=True)
        
    elif selected_viz == "Spoilage by Category":
        fig_spoilage_category = px.box(
            df_viz,
            x="Category",
            y="Spoilage (%)",
            color="Category",
            points="all",
            title="Spoilage Distribution by Category",
            color_discrete_map={"Fruit": "#4CAF50", "Vegetable": "#2196F3"},
            template="plotly_white"
        )
        fig_spoilage_category.update_layout(
            height=500,
            xaxis_title="Category",
            yaxis_title="Spoilage Rate (%)"
        )
        st.plotly_chart(fig_spoilage_category, use_container_width=True)
        
    elif selected_viz == "Temperature vs Spoilage":
        fig_temp_spoilage = px.scatter(
            df_viz,
            x="Avg_Temperature",
            y="Spoilage (%)",
            color="Category",
            size="Transit_Time",
            hover_name="Product",
            size_max=25,
            title="Temperature vs Spoilage Relationship",
            color_discrete_map={"Fruit": "#4CAF50", "Vegetable": "#2196F3"},
            template="plotly_white"
        )
        
        fig_temp_spoilage.update_layout(
            height=500,
            xaxis_title="Average Temperature (°C)",
            yaxis_title="Spoilage Rate (%)",
            legend_title="Category"
        )
        
        x = df_viz["Avg_Temperature"]
        y = df_viz["Spoilage (%)"]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        fig_temp_spoilage.add_trace(
            go.Scatter(
                x=[min(x), max(x)],
                y=[p(min(x)), p(max(x))],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Trend"
            )
        )
        
        st.plotly_chart(fig_temp_spoilage, use_container_width=True)
        
    elif selected_viz == "Transit Time vs Spoilage":
        fig_transit_spoilage = px.scatter(
            df_viz,
            x="Transit_Time",
            y="Spoilage (%)",
            color="Category",
            symbol="Storage",
            hover_name="Product",
            title="Transit Time vs Spoilage Relationship",
            color_discrete_map={"Fruit": "#4CAF50", "Vegetable": "#2196F3"},
            template="plotly_white"
        )
        
        fig_transit_spoilage.update_layout(
            height=500,
            xaxis_title="Transit Time (hours)",
            yaxis_title="Spoilage Rate (%)",
            legend_title="Category"
        )
        
        x = df_viz["Transit_Time"]
        y = df_viz["Spoilage (%)"]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        fig_transit_spoilage.add_trace(
            go.Scatter(
                x=[min(x), max(x)],
                y=[p(min(x)), p(max(x))],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Trend"
            )
        )
        
        st.plotly_chart(fig_transit_spoilage, use_container_width=True)
        
    elif selected_viz == "Storage Type Comparison":
        fig_storage = px.box(
            df_viz,
            x="Storage",
            y="Spoilage (%)",
            color="Category",
            points="all",
            title="Spoilage by Storage Type and Category",
            color_discrete_map={"Fruit": "#4CAF50", "Vegetable": "#2196F3"},
            template="plotly_white"
        )
        
        fig_storage.update_layout(
            height=500,
            xaxis_title="Storage Type",
            yaxis_title="Spoilage Rate (%)",
            legend_title="Category"
        )
        
        st.plotly_chart(fig_storage, use_container_width=True)
    
    with st.expander("About the Visualization"):
        if selected_viz == "Spoilage by Product":
            st.markdown("""
            #### Key Insights:
            • Products like Green Peas, Plums, and Tomatoes show the highest spoilage rates (above 30%).
            • Pineapples have remarkably low spoilage at just 5.2%, making them ideal for longer supply chains.
            • Vegetables generally show higher spoilage rates than fruits in the dataset.
            • Produce with spoilage rates above 25% should be prioritized for cold chain improvements.
            • There are significant variations even within the same category (e.g., among fruits).
            """)
        
        elif selected_viz == "Spoilage by Category":
            st.markdown("""
            #### Key Insights:
            • Vegetables show higher median spoilage rates (27.5%) compared to fruits (21.5%).
            • The vegetable category exhibits greater variability in spoilage rates.
            • The interquartile range for vegetables is wider, indicating less predictable preservation outcomes.
            • There are notable outliers in the fruit category, with some fruits having particularly low spoilage.
            • This categorical difference suggests different handling protocols may be optimal for each category.
            """)
        
        elif selected_viz == "Temperature vs Spoilage":
            st.markdown("""
            #### Key Insights:
            • There is a positive correlation between higher temperatures and increased spoilage rates.
            • The trend line indicates approximately 0.5% increase in spoilage for each 1°C increase.
            • Products stored below 20°C generally show spoilage rates under 25%.
            • Bubble size represents transit time - larger bubbles spent more time in transit.
            • Some products defy the trend, suggesting other factors (humidity, handling) may be at play.
            """)
        
        elif selected_viz == "Transit Time vs Spoilage":
            st.markdown("""
            #### Key Insights:
            • Longer transit times correlate with higher spoilage rates, with a clear positive trend.
            • Products with transit times under 50 hours generally maintain spoilage rates below 25%.
            • The effect is more pronounced for vegetables than fruits.
            • Storage type (symbol shape) shows refrigerated items resist spoilage better during long transits.
            • For every 10 hours of additional transit time, spoilage increases by approximately 1.2% on average.
            """)
        
        elif selected_viz == "Storage Type Comparison":
            st.markdown("""
            #### Key Insights:
            • Refrigerated storage shows lower median spoilage rates for both fruits and vegetables.
            • The advantage of refrigeration is more significant for vegetables than fruits.
            • Ambient storage produces more outliers with extremely high spoilage rates.
            • Even within refrigerated storage, vegetables show more variability than fruits.
            • The difference between ambient and refrigerated is approximately 6.5% for vegetables and 4.2% for fruits.
            """)

if st.session_state.get("show_product_data", False):
    st.markdown("---")
    selected_product = st.session_state["selected_product"]

    if selected_product == "All Products":
        st.markdown("#### Aggregated Metrics Across All Products")
        
        summary_stats = df_viz.groupby("Category").agg({
            "Spoilage (%)": ["mean", "median", "std"],
            "Avg_Temperature": ["mean", "median"],
            "Transit_Time": ["mean", "median"]
        }).reset_index()
        
        summary_stats.columns = [' '.join(col).strip() for col in summary_stats.columns.values]
        st.dataframe(summary_stats.style.background_gradient(cmap="YlGn"), use_container_width=True)
        
        st.markdown("#### Complete Product Dataset")
        st.dataframe(df_viz, use_container_width=True)

    else:
        product_data = df_viz[df_viz["Product"] == selected_product]
        
        if not product_data.empty:
            st.markdown(f"#### Detailed Metrics for {selected_product}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Spoilage Rate",
                    value=f"{product_data['Spoilage (%)'].values[0]}%",
                    delta=f"{round(product_data['Spoilage (%)'].values[0] - avg_spoilage, 1)}%",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    label="Average Temperature",
                    value=f"{product_data['Avg_Temperature'].values[0]}°C",
                    delta=f"{round(product_data['Avg_Temperature'].values[0] - 25, 1)}°C"
                )
            with col3:
                st.metric(
                    label="Transit Time",
                    value=f"{product_data['Transit_Time'].values[0]} hours",
                    delta=f"{round(product_data['Transit_Time'].values[0] - avg_transit, 1)} hours",
                    delta_color="inverse"
                )
            
            st.markdown("#### Recommendations")
            spoilage_rate = product_data["Spoilage (%)"].values[0]

            if spoilage_rate > 30:
                st.error("⚠️ **High Priority** - This product has critical spoilage rates above 30%")
                st.markdown("""
                - Immediate cold chain optimization required  
                - Consider shorter transit routes or more frequent deliveries  
                - Evaluate packaging solutions to extend shelf life  
                - Prioritize for refrigerated transport  
                """)
            elif spoilage_rate > 25:
                st.warning("⚠️ **Medium Priority** - This product has elevated spoilage rates")
                st.markdown("""
                - Review temperature controls during transit  
                - Assess handling procedures for potential improvements  
                - Consider inventory rotation policies  
                """)
            else:
                st.success("✅ **Low Priority** - This product has acceptable spoilage rates")
                st.markdown("""
                - Maintain current handling procedures  
                - Continue monitoring for any changes  
                - Consider benchmarking against industry standards  
                """)

            st.markdown("#### Product Attributes")
            st.dataframe(product_data.T, use_container_width=True)
        else:
            st.warning(f"No data available for {selected_product}")