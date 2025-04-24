# Supply Chain Bottleneck Detection Dashboard

A comprehensive data visualization and machine learning application for identifying, analyzing, and predicting bottlenecks in supply chain operations.

## Features

- **Real-time Bottleneck Prediction**: Machine learning model to predict bottlenecks in your supply chain
- **Top Vulnerable Routes**: Identifies the 5 most vulnerable routes in your logistics network
- **Critical Route Focus**: Automatically analyzes the most critical route (highest risk score) for detailed insights
- **Route Selection Analysis**: Dropdown selector to choose and analyze any of the top 5 critical routes with detailed risk factors and recommendations
- **Interactive Visualizations**: Multiple charts to understand patterns and trends
- **Department Performance**: Compare how different departments are performing
- **Risk Score Calculation**: Custom risk scoring algorithm based on multiple factors
- **Data Export**: Save your analysis results or download the processed data
- **Modern UI**: Sleek interface with online images and responsive design

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

1. Upload your supply chain data in CSV format or use the provided sample data
2. The application will automatically analyze the data and predict bottlenecks
3. The most critical route (highest risk score) will be highlighted for immediate attention
4. Navigate through different tabs to explore various aspects of your supply chain
5. Review the Top 5 Vulnerable Routes table to identify other high-risk areas
6. Use the dropdown selector to analyze any of the top 5 critical routes in detail
7. Review the risk factors and recommendations for the selected route
8. Save results or download processed data using the sidebar options

## Data Format

Your CSV data should include the following columns:
- department: The department handling the route
- time: Timestamp of the shipment/operation
- resource_usage: Resource utilization percentage (0-100)
- utility_allocation: Utility allocation percentage (0-100)
- priority: Priority level (Low, Medium, High)
- departure_loc: Starting location
- arrival_loc: Destination location
- deadline: Target completion date
- return_status: Whether returns are expected (Yes/No)
- festive_season: Whether it's during a festive period (Yes/No)
- shelf_life: Product shelf life in days
- status: Current status (Safe or Bottleneck) for training data

## Included Datasets

The application comes with three datasets for different purposes:

1. **data.csv** - Main training dataset with standard supply chain routes
2. **test.csv** - Test dataset with different routes but similar patterns to validate model performance
3. **validation.csv** - Validation dataset with edge cases and unusual values to test model robustness

To use these datasets:
1. Start the application with `streamlit run main.py`
2. Select "Upload My Data" in the sidebar
3. Upload one of the provided CSV files

## Customization

You can modify the risk scoring algorithm in the code to adjust weights according to your business priorities.

You can also customize the application's appearance:
- Replace the logo images by changing the URLs in the markdown sections
- Modify the CSS in the markdown section to change colors, fonts, and layout
- Add additional visualization options by extending the charts section

## Model Evaluation

After testing with different datasets, you can evaluate the model's performance by:

1. Comparing the predicted bottlenecks with actual bottlenecks
2. Reviewing the feature importance chart to understand key factors
3. Analyzing the top vulnerable routes against your domain knowledge
4. Testing with the validation dataset to see how the model handles edge cases

## Working with the Validation Dataset

The validation dataset (`validation.csv`) contains several edge cases designed to test the robustness of the model:

1. **Extreme Values**: Contains 0% and 100% values for resource usage and utility allocation
2. **New Departments**: Includes departments not in the training data (Sales, Marketing, etc.)
3. **Unusual Combinations**: Tests scenarios like high resource usage with high utility allocation
4. **Shelf Life Extremes**: Ranges from 1 day to 150 days

When working with this dataset, the application will:
- Map new departments to standard departments for model compatibility
- Handle extreme values through risk score clipping
- Provide alternative visualizations when standard charts cannot be generated
- Display appropriate warnings about unusual data patterns

## Troubleshooting

If you encounter errors when using the application, try the following:

1. **Data Format Issues**: Ensure your CSV has all required columns with appropriate data types
2. **Missing Values**: The application handles missing values, but too many might affect model performance
3. **New Categories**: The application can handle new categorical values but works best with standard ones
4. **Visualization Errors**: Alternative visualizations will be shown when standard ones cannot be generated
5. **Model Training Failures**: The application will attempt to use a previously trained model or provide limited functionality

## Future Enhancements

Potential enhancements for future versions:

1. **Model Comparison**: Adding multiple model options to compare performance
2. **Time Series Forecasting**: Predicting future bottlenecks based on historical patterns
3. **Interactive Map Visualization**: Geographical representation of supply chain routes
4. **Custom Risk Scoring**: User interface for adjusting risk score weights
5. **Anomaly Detection**: Identifying unusual patterns in the supply chain data 