
# 🌞 Solar Energy Loss Analysis - Advanced ML Application

A comprehensive Streamlit application for analyzing solar PV plant performance using advanced machine learning techniques to quantify and attribute energy losses across multiple temporal and system levels.

[![Watch Demo Video](https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/yZMfwE-iFUI?si=9inxjXMfDt5VcOy-)

## 📋 Overview

This application provides sophisticated analysis of solar PV plant performance by implementing state-of-the-art machine learning pipelines for theoretical generation estimation and granular loss attribution methodology. The system analyzes energy losses across five key categories:

- **☁️ Cloud Cover** - Energy losses due to cloud attenuation
- **🌑 Shading** - Losses from static/dynamic objects blocking sunlight  
- **🌡️ Temperature Effects** - Losses due to high module/inverter temperatures
- **🧹 Soiling** - Energy reduction due to dust and dirt accumulation
- **❓ Other/Novel Losses** - Unexplored or residual loss factors

## 🚀 Quick Start

### 1. Train the Model (Required First Step)
```bash
python train_model.py
```

### 2. Launch the Application
```bash
streamlit run main_app.py --server.port 5000
```

### 3. Access the Application
Open your browser to the provided URL (typically `http://0.0.0.0:5000` or `https://localhost:5000`)

## 📁 Required Files

### Core Application Files
- **`main_app.py`** - Main Streamlit application with navigation and UI
- **`data_processor.py`** - Data loading, preprocessing, and feature engineering
- **`train_model.py`** - Model training script (run this first)
- **`model_deployment.py`** - Model loading and deployment utilities
- **`loss_attribution.py`** - Loss attribution methodology implementation
- **`visualization.py`** - Interactive visualizations and dashboards
- **`utils.py`** - Utility functions and helper methods

### Data & Configuration
- **`data/data.csv`** - Primary dataset (17,000+ solar plant measurements)
- **`.streamlit/config.toml`** - Streamlit configuration

### Auto-Generated (After Training)
- **`models/`** - Directory containing trained model artifacts:
  - `best_theoretical_model.pkl` - Trained ML model
  - `data_processor.pkl` - Data preprocessing pipeline
  - `model_metadata.pkl` - Model performance metrics and metadata

## 🏗️ System Architecture

### Advanced ML Pipeline
- **Gradient Boosting Models**: LightGBM, XGBoost, Random Forest
- **Ensemble Methods**: Voting regressor for improved accuracy
- **Cross-Validation**: TimeSeriesSplit for temporal data integrity
- **Feature Engineering**: 260+ engineered features including solar position calculations

### Multi-Granularity Analysis
- **Temporal**: 15-minute, hourly, daily, weekly, monthly analysis
- **System Levels**: Plant-level, Inverter-level (INV-3, INV-8), String-level
- **Interactive Filtering**: Date range, asset, and granularity selection

### Loss Attribution Methodology
- **Scientific Approach**: Statistically sound loss quantification
- **Residual Calculation**: Other losses = Total - (Cloud + Shading + Temperature + Soiling)
- **Validation**: Ensures attributed losses sum to total measured loss

## 📊 Plant Specifications

- **Total Capacity**: 7.6 MW (2 × 3.8 MW inverters)
- **Location**: 38°0'2"N, 1°20'4"W (Spain)
- **Inverters**: INV-3 (CTIN03), INV-8 (CTIN08)
- **Data Resolution**: 15-minute intervals
- **Analysis Period**: Full dataset with 17,000+ measurements

## 🎯 Key Features

### 1. **Data Exploration**
- Comprehensive dataset overview and statistics
- Data quality assessment and missing value analysis
- Time series visualization and distribution analysis

### 2. **Theoretical Generation Model**
- Pre-trained ML model for baseline generation estimation
- Model performance metrics and feature importance
- Cross-validation results and comparison

### 3. **Loss Attribution Analysis**
- Interactive loss breakdown by category and time period
- Multi-level analysis (Plant/Inverter/String)
- Real-time loss calculation and visualization

### 4. **Asset Performance Ranking**
- Inverter and string performance comparison
- Efficiency metrics and loss distribution
- Performance trend analysis

### 5. **Insights & Recommendations**
- Operational optimization suggestions
- Maintenance priority identification
- Strategic improvement recommendations

## 🔧 Technical Requirements

- **Python 3.8+**
- **Streamlit** (automatically installed)
- **Machine Learning Libraries**: scikit-learn, lightgbm, xgboost
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn

All dependencies are automatically managed by the Replit environment.

## 📈 Expected Results

### Model Performance
- **R² Score**: Typically >0.90 for theoretical generation prediction
- **Feature Importance**: Solar position and irradiance as top predictors
- **Cross-Validation**: Robust performance across time periods

### Loss Attribution
- **Cloud Cover**: ~35% of total losses
- **Temperature Effects**: ~28% of total losses
- **Soiling**: ~15% of total losses
- **Shading**: ~12% of total losses
- **Other Factors**: ~10% of total losses

## 🚦 Troubleshooting

### Missing Model Files Error
If you see "Missing model files" in the console:
1. Stop the application (Ctrl+C)
2. Run the training script: `python train_model.py`
3. Wait for training to complete (5-15 minutes)
4. Restart the application: `streamlit run main_app.py --server.port 5000`

### Data Processing Issues
If dataset is reduced to only 5 records:
- Check data file integrity in `data/data.csv`
- Verify datetime column format
- Review preprocessing logs for filtering details

### WebSocket Disconnections
Multiple "WebSocket onclose" messages indicate network connectivity issues but don't affect core functionality.

## 🎯 Next Steps

1. **Training**: Always run `train_model.py` first
2. **Analysis**: Use the Streamlit interface for comprehensive analysis
3. **Optimization**: Implement recommendations from the insights section
4. **Monitoring**: Set up regular model retraining for optimal performance

## 📧 Support

For technical issues or questions about the implementation, refer to the methodology section within the application for detailed explanations of algorithms and assumptions.
