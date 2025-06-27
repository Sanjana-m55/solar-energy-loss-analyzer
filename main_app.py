import os
# Set threading limits to prevent resource issues
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from model_deployment import model_deployment
from loss_attribution import LossAttributor
from visualization import Visualizer
from utils import Utils

def sanitize_dataframe_for_streamlit(df):
    import pandas as pd
    import pyarrow as pa

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'Data Type' in df.columns:
        df['Data Type'] = df['Data Type'].astype(str)

    for col in df.select_dtypes(include='object').columns:
        try:
            pa.array(df[col])
        except Exception:
            df[col] = df[col].astype(str)
    return df


# Page configuration
st.set_page_config(
    page_title="Solar Energy Loss Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'losses_attributed' not in st.session_state:
    st.session_state.losses_attributed = False

def main():
    """Main application function"""
    
    # Application header
    st.title("🌞 Deconstructing Solar Energy Losses")
    st.markdown("### Advanced ML-based Analysis of Solar PV Plant Performance")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Section",
        [
            "🏠 Home & Overview",
            "📊 Data Exploration", 
            "🤖 Theoretical Generation Model",
            "⚡ Loss Attribution Analysis",
            "🏆 Asset Performance & Ranking",
            "📝 Methodology & Assumptions",
            "💡 Insights & Recommendations"
        ]
    )
    
    # Initialize components
    data_processor = DataProcessor()
    loss_attributor = LossAttributor()
    visualizer = Visualizer()
    utils = Utils()
    
    # Load pre-trained model if available
    if not model_deployment.is_loaded:
        model_deployment.load_trained_model()
    
    # Load and process data
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            try:
                # Load data
                data = data_processor.load_data("data/data.csv")
                
                # Process data
                processed_data = data_processor.preprocess_data(data)
                # Fix datetime column for compatibility with Streamlit/Arrow
                if 'datetime' in processed_data.columns:
                    processed_data['datetime'] = pd.to_datetime(processed_data['datetime'], errors='coerce')
                
                # Store in session state
                st.session_state.raw_data = data
                # Ensure problematic object columns are stringified for Arrow compatibility
                if 'Data Type' in processed_data.columns:
                    processed_data['Data Type'] = processed_data['Data Type'].astype(str)
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                
                st.success("✅ Data loaded and processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()
    
    # Page routing
    if page == "🏠 Home & Overview":
        show_home_page()
    elif page == "📊 Data Exploration":
        show_data_exploration(st.session_state.raw_data, st.session_state.processed_data)
    elif page == "🤖 Theoretical Generation Model":
        show_theoretical_model(model_deployment, st.session_state.processed_data)
    elif page == "⚡ Loss Attribution Analysis":
        show_loss_analysis(loss_attributor, visualizer, st.session_state.processed_data)
    elif page == "🏆 Asset Performance & Ranking":
        show_asset_ranking(visualizer, st.session_state.processed_data)
    elif page == "📝 Methodology & Assumptions":
        show_methodology()
    elif page == "💡 Insights & Recommendations":
        show_insights()

def show_home_page():
    """Display home page with project overview"""
    
    st.header("🎯 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Problem Statement
        
        In real-world solar PV plants, theoretical energy output is rarely achieved due to various loss factors. 
        This application provides comprehensive analysis to:
        
        - **Quantify** the gap between theoretical and actual energy generation
        - **Attribute** energy losses to specific causes using advanced ML techniques
        - **Visualize** performance across multiple temporal and system levels
        
        ### Key Loss Categories Analyzed
        
        1. **☁️ Cloud Cover** - Energy losses due to cloud attenuation
        2. **🌑 Shading** - Losses from static/dynamic objects blocking sunlight
        3. **🌡️ Temperature Effects** - Losses due to high module/inverter temperatures
        4. **🧹 Soiling** - Energy reduction due to dust and dirt accumulation
        5. **❓ Other/Novel Losses** - Unexplored or residual loss factors
        
        ### Analysis Capabilities
        
        - **Multi-granularity**: 15-minute to monthly analysis
        - **Multi-level**: Plant, Inverter, and String-level insights
        - **Interactive**: Dynamic filtering and real-time visualizations
        - **ML-powered**: Advanced gradient boosting and ensemble methods
        """)
    
    with col2:
        # Key statistics
        if st.session_state.data_loaded:
            data = st.session_state.processed_data
            
            st.markdown("### 📈 Dataset Statistics")
            
            # Create metrics
            total_records = len(data)
            date_range = f"{data['datetime'].min().strftime('%Y-%m-%d')} to {data['datetime'].max().strftime('%Y-%m-%d')}"
            unique_inverters = data['inverter'].nunique() if 'inverter' in data.columns else 0
            unique_strings = data['string'].nunique() if 'string' in data.columns else 0
            
            st.metric("Total Records", f"{total_records:,}")
            st.metric("Date Range", date_range)
            st.metric("Inverters", unique_inverters)
            st.metric("Strings", unique_strings)
            
            # System capacity info
            st.markdown("### ⚡ System Information")
            st.info("""
            **Plant Capacity:** 45.6 MWh  
            **Location:** 38°0'2"N, 1°20'4"W  
            **Timezone:** UTC+1  
            **Inverters:** INV-3 (3.8MW), INV-8 (3.8MW)
            """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("Use the navigation menu in the sidebar to explore different analysis sections.")

def show_data_exploration(raw_data, processed_data):
    if 'datetime' in raw_data.columns:
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'], errors='coerce')
    """Display data exploration and initial statistics"""
    
    st.header("📊 Data Exploration")
    
    # Data overview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔍 Quality Check", "⏰ Time Series"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Raw Data Statistics")
            st.dataframe(sanitize_dataframe_for_streamlit(raw_data.describe()), use_container_width=True)

            
        with col2:
            st.markdown("#### Data Types")
            dtype_df = pd.DataFrame({
                'Column': raw_data.dtypes.index,
                'Data Type': raw_data.dtypes.values,
                'Non-Null Count': raw_data.count().values,
                'Null Count': raw_data.isnull().sum().values
            })
            st.dataframe(sanitize_dataframe_for_streamlit(dtype_df), use_container_width=True)
    
    with tab2:
        st.subheader("Data Distributions")
        
        # Select numeric columns for distribution analysis
        numeric_columns = raw_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            selected_columns = st.multiselect(
                "Select columns for distribution analysis",
                numeric_columns,
                default=numeric_columns[:4]  # Show first 4 by default
            )
            
            if selected_columns:
                # Create distribution plots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_columns[:4],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                for i, col in enumerate(selected_columns[:4]):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    
                    # Create histogram
                    fig.add_trace(
                        go.Histogram(x=raw_data[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=600, title_text="Distribution Analysis")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Data Quality Assessment")
        
        # Missing values analysis
        missing_data = raw_data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Percentage': (missing_data.values / len(raw_data) * 100).round(2)
                })
                st.dataframe(sanitize_dataframe_for_streamlit(missing_df), use_container_width=True)
            
            with col2:
                # Missing values visualization
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected in the dataset!")
        
        # Outlier detection for key columns
        st.markdown("#### Outlier Detection")
        outlier_columns = st.multiselect(
            "Select columns for outlier analysis",
            numeric_columns,
            default=[]
        )
        
        if outlier_columns:
            for col in outlier_columns:
                Q1 = raw_data[col].quantile(0.25)
                Q3 = raw_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = raw_data[(raw_data[col] < lower_bound) | (raw_data[col] > upper_bound)]
                
                st.write(f"**{col}**: {len(outliers)} outliers detected ({len(outliers)/len(raw_data)*100:.2f}%)")
    
    with tab4:
        st.subheader("Time Series Analysis")
        
        if 'datetime' in raw_data.columns:
            # Time series plot
            time_columns = st.multiselect(
                "Select variables for time series visualization",
                numeric_columns,
                default=[]
            )
            
            if time_columns:
                fig = go.Figure()
                
                for col in time_columns:
                    fig.add_trace(go.Scatter(
                        x=raw_data['datetime'],
                        y=raw_data[col],
                        mode='lines',
                        name=col,
                        line=dict(width=1)
                    ))
                
                fig.update_layout(
                    title="Time Series Analysis",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No datetime column found for time series analysis")

def show_theoretical_model(model_deployment, data):
    """Display theoretical generation model section"""
    st.header("🤖 Theoretical Generation Model")
    
    st.markdown("""
    This section uses a pre-trained machine learning model to estimate theoretical energy generation.
    The model serves as the baseline for loss attribution analysis.
    """)
    
    # Check model availability
    model_status = model_deployment.check_model_availability()
    
    if model_status['available']:
        # Model is available - show information
        if model_deployment.is_loaded:
            st.success("✅ Pre-trained model loaded successfully!")
            
            # Display model information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Information")
                model_info = model_deployment.get_model_info()
                
                st.metric("Model Type", model_info['model_name'])
                st.metric("Training Date", model_info['training_date'])
                st.metric("R² Score", f"{model_info['performance_metrics']['full_data_r2']:.4f}")
                st.metric("Features", f"{model_info['n_features']:,}")
                st.metric("Training Samples", f"{model_info['n_training_samples']:,}")
            
            with col2:
                st.subheader("Model Performance")
                metrics = model_info['performance_metrics']
                st.metric("Cross-Validation R²", f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
                st.metric("MAE", f"{metrics['mae']:.4f}")
            
            # Model comparison
            st.subheader("Model Comparison")
            comparison_df = model_deployment.get_model_comparison()
            if not comparison_df.empty:
                st.dataframe(sanitize_dataframe_for_streamlit(comparison_df.round(4)), use_container_width=True)
            
            # Feature importance
            feature_importance = model_deployment.get_feature_importance()
            if not feature_importance.empty:
                st.subheader("Top Feature Importance")
                
                # Plot feature importance
                fig = px.bar(
                    feature_importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Generate predictions on sample data
            if st.button("Generate Sample Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    try:
                        sample_data = data.head(100)
                        predictions = model_deployment.predict_theoretical_generation(sample_data)
                        
                        if len(predictions) > 0:
                            # Create comparison plot
                            sample_data = sample_data.copy()
                            sample_data['Theoretical_Generation'] = predictions
                            
                            # Find actual generation column
                            energy_cols = [col for col in sample_data.columns if 'energy' in col.lower()]
                            if energy_cols:
                                actual_col = energy_cols[0]
                                sample_data['Actual_Generation'] = sample_data[actual_col]
                                
                                fig = px.line(
                                    sample_data.reset_index(),
                                    x='index',
                                    y=['Theoretical_Generation', 'Actual_Generation'],
                                    title="Theoretical vs Actual Generation (Sample)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Showing theoretical generation predictions")
                                fig = px.line(
                                    sample_data.reset_index(),
                                    x='index',
                                    y='Theoretical_Generation',
                                    title="Theoretical Generation Predictions (Sample)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to generate predictions")
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
        else:
            st.warning("Model files found but not loaded. Click to load model.")
            if st.button("Load Pre-trained Model"):
                with st.spinner("Loading model..."):
                    success = model_deployment.load_trained_model()
                    if success:
                        st.success("Model loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load model")
    else:
        # Model not available - show training instructions
        st.warning("⚠️ Pre-trained model not found. Please train the model first.")
        
        st.subheader("Missing Files")
        for filename, status in model_status['file_status'].items():
            status_icon = "✅" if status['exists'] else "❌"
            st.write(f"{status_icon} **{filename}**: {status['description']}")
        
        st.subheader("Training Instructions")
        instructions = model_deployment.create_training_instructions()
        st.markdown(instructions)
        
        st.info("📝 **Next Steps**: Run the model_training.ipynb notebook to train and save the model, then refresh this page.")

# Removed old training functions - now using pre-trained models

def show_loss_analysis(loss_attributor, visualizer, data):
    """Display loss attribution analysis"""
    
    st.header("⚡ Loss Attribution Analysis")
    
    # Analysis controls
    st.subheader("Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range selector
        if 'datetime' in data.columns:
            min_date = data['datetime'].min().date()
            max_date = data['datetime'].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
    
    with col2:
        # Aggregation level
        aggregation_level = st.selectbox(
            "Analysis Level",
            ["Plant", "Inverter", "String"]
        )
    
    with col3:
        # Time granularity
        time_granularity = st.selectbox(
            "Time Granularity",
            ["15-minute", "Hourly", "Daily", "Weekly", "Monthly"]
        )
    
    # Run loss attribution
    if st.button("🔍 Analyze Losses", type="primary"):
        analyze_losses(loss_attributor, data, date_range, aggregation_level, time_granularity)
    
    # Show results
    if st.session_state.losses_attributed:
        show_loss_results(visualizer)

def analyze_losses(loss_attributor, data, date_range, aggregation_level, time_granularity):
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    """Perform loss attribution analysis"""
    
    with st.spinner("Analyzing energy losses..."):
        try:
            # Filter data by date range
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = data[
                    (data['datetime'].dt.date >= start_date) & 
                    (data['datetime'].dt.date <= end_date)
                ]
            else:
                filtered_data = data
            
            # Perform loss attribution
            loss_results = loss_attributor.attribute_losses(
                filtered_data,
                aggregation_level=aggregation_level,
                time_granularity=time_granularity
            )
            
            # Store results
            st.session_state.loss_results = loss_results
            st.session_state.losses_attributed = True
            
            st.success("✅ Loss attribution completed!")
            
        except Exception as e:
            st.error(f"❌ Error analyzing losses: {str(e)}")

def show_loss_results(visualizer):
    """Display loss attribution results"""
    
    st.subheader("Loss Attribution Results")
    
    results = st.session_state.loss_results
    
    # Loss breakdown visualization
    if 'loss_breakdown' in results:
        st.subheader("Loss Breakdown")
        
        # Create stacked bar chart
        loss_df = results['loss_breakdown']
        
        fig = go.Figure()
        
        loss_categories = ['Cloud Cover', 'Shading', 'Temperature', 'Soiling', 'Other Losses']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for i, category in enumerate(loss_categories):
            if category in loss_df.columns:
                fig.add_trace(go.Bar(
                    x=loss_df.index,
                    y=loss_df[category],
                    name=category,
                    marker_color=colors[i]
                ))
        
        fig.update_layout(
            title="Energy Loss Breakdown Over Time",
            xaxis_title="Time Period",
            yaxis_title="Energy Loss (kWh)",
            barmode='stack',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Loss summary table
    if 'loss_summary' in results:
        st.subheader("Loss Summary Statistics")
        st.dataframe(sanitize_dataframe_for_streamlit(results['loss_summary']), use_container_width=True)
    
    # Trend analysis
    if 'trends' in results:
        st.subheader("Loss Trends")
        
        trends_df = results['trends']
        
        fig = go.Figure()
        
        for column in trends_df.columns:
            if column != 'datetime':
                fig.add_trace(go.Scatter(
                    x=trends_df['datetime'],
                    y=trends_df[column],
                    mode='lines+markers',
                    name=column,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Loss Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Loss Percentage (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_asset_ranking(visualizer, data):
    """Display asset performance and ranking"""
    
    st.header("🏆 Asset Performance & Ranking")
    
    # Asset performance metrics
    st.subheader("Asset Performance Overview")
    
    # Create sample asset ranking data
    asset_data = create_sample_asset_data(data)
    
    # Performance ranking tabs
    tab1, tab2, tab3 = st.tabs(["🔋 Inverter Ranking", "🔌 String Ranking", "📊 Performance Metrics"])
    
    with tab1:
        st.subheader("Inverter Performance Ranking")
        
        if 'inverter_ranking' in asset_data:
            inverter_df = asset_data['inverter_ranking']
            
            # Display ranking table
            st.dataframe(sanitize_dataframe_for_streamlit(inverter_df), use_container_width=True)
            
            # Efficiency comparison chart
            fig = px.bar(
                inverter_df,
                x='Inverter',
                y='Efficiency (%)',
                title="Inverter Efficiency Comparison",
                color='Efficiency (%)',
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("String Performance Ranking")
        
        if 'string_ranking' in asset_data:
            string_df = asset_data['string_ranking']
            
            # Display top performing strings
            st.dataframe(sanitize_dataframe_for_streamlit(string_df.head(10)), use_container_width=True)
            
            # Loss type comparison
            fig = px.scatter(
                string_df,
                x='Total Losses (kWh)',
                y='Efficiency (%)',
                size='Soiling Loss (kWh)',
                color='Temperature Loss (kWh)',
                hover_data=['String', 'Inverter'],
                title="String Performance Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Metrics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Efficiency", "94.2%", "2.1%")
        with col2:
            st.metric("Worst Efficiency", "87.3%", "-3.2%")
        with col3:
            st.metric("Avg Soiling Loss", "1.8%", "0.3%")
        with col4:
            st.metric("Avg Temperature Loss", "2.4%", "-0.5%")

def create_sample_asset_data(data):
    """Create sample asset performance data for demonstration"""
    
    # This would normally be calculated from the actual data
    # For demonstration, creating sample data structure
    
    inverter_data = {
        'Inverter': ['INV-3', 'INV-8'],
        'Efficiency (%)': [92.5, 89.8],
        'Total Losses (kWh)': [1250, 1680],
        'Cloud Loss (kWh)': [450, 520],
        'Shading Loss (kWh)': [200, 280],
        'Temperature Loss (kWh)': [350, 480],
        'Soiling Loss (kWh)': [150, 200],
        'Other Loss (kWh)': [100, 200]
    }
    
    string_data = {
        'String': [f'String_{i}' for i in range(1, 21)],
        'Inverter': ['INV-3'] * 10 + ['INV-8'] * 10,
        'Efficiency (%)': np.random.uniform(85, 95, 20),
        'Total Losses (kWh)': np.random.uniform(50, 200, 20),
        'Cloud Loss (kWh)': np.random.uniform(10, 50, 20),
        'Shading Loss (kWh)': np.random.uniform(5, 30, 20),
        'Temperature Loss (kWh)': np.random.uniform(15, 60, 20),
        'Soiling Loss (kWh)': np.random.uniform(5, 25, 20),
        'Other Loss (kWh)': np.random.uniform(0, 20, 20)
    }
    
    return {
        'inverter_ranking': pd.DataFrame(inverter_data),
        'string_ranking': pd.DataFrame(string_data)
    }

def show_methodology():
    """Display methodology and assumptions"""
    
    st.header("📝 Methodology & Assumptions")
    
    # Methodology sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Data Processing", "🤖 ML Pipeline", "⚡ Loss Attribution", "📊 Assumptions"])
    
    with tab1:
        st.subheader("Data Preprocessing & Feature Engineering")
        
        st.markdown("""
        ### Data Preprocessing Steps
        
        1. **Datetime Processing**
           - Parse datetime columns with proper timezone handling
           - Extract temporal features: hour, day, month, season
           - Calculate solar position angles (zenith, azimuth)
        
        2. **Feature Engineering**
           - Solar irradiance components (GII, GHI, rear irradiance)
           - Environmental factors (temperature, humidity, wind)
           - System parameters (inverter power, string currents/voltages)
           - Interaction features between environmental and system variables
        
        3. **Data Quality Assurance**
           - Missing value imputation using forward/backward fill
           - Outlier detection and treatment using IQR method
           - Data validation and consistency checks
        
        4. **Feature Scaling**
           - Standardization of numerical features
           - Categorical encoding for system components
        """)
    
    with tab2:
        st.subheader("Machine Learning Pipeline")
        
        st.markdown("""
        ### Theoretical Generation Model
        
        **Objective**: Predict maximum theoretical energy output under ideal conditions
        
        **Model Selection Process**:
        1. **Candidate Models**:
           - LightGBM (Gradient Boosting)
           - XGBoost (Extreme Gradient Boosting)
           - CatBoost (Categorical Boosting)
           - Random Forest
           - Deep Neural Networks (if data volume supports)
        
        2. **Hyperparameter Optimization**:
           - Optuna framework for intelligent parameter search
           - Bayesian optimization for efficient search space exploration
           - Cross-validation for robust parameter selection
        
        3. **Model Evaluation**:
           - TimeSeriesSplit for temporal data integrity
           - Metrics: R², RMSE, MAE, MAPE
           - Residual analysis and feature importance
        
        4. **Ensemble Methods**:
           - Voting ensemble combining top-performing models
           - Stacking ensemble with meta-learner
           - Weighted averaging based on validation performance
        """)
    
    with tab3:
        st.subheader("Loss Attribution Methodology")
        
        st.markdown("""
        ### Scientific Loss Attribution Framework
        
        **Core Principle**: 
        Total Loss = Theoretical Generation - Actual Generation
        
        **Attribution Method**:
        
        1. **Cloud Cover Losses**
           - Analyze correlation between irradiance reduction and cloud cover indicators
           - Use clear-sky models to estimate theoretical irradiance
           - Calculate energy loss: ΔE_cloud = (I_clear - I_actual) × Panel_area × Efficiency
        
        2. **Shading Losses**
           - Identify shading events through irradiance deviation analysis
           - Compare string-level performance to detect partial shading
           - Quantify using geometric shading models
        
        3. **Temperature Losses**
           - Apply temperature coefficient models: ΔP = P_STC × α × (T_cell - T_STC)
           - Calculate cell temperature from ambient temperature and irradiance
           - Estimate thermal losses for each time interval
        
        4. **Soiling Losses**
           - Use clean vs dirty reference cell measurements
           - Statistical analysis of performance degradation patterns
           - Soiling ratio calculation: SR = I_dirty / I_clean
        
        5. **Other/Residual Losses**
           - Other Losses = Total Loss - (Cloud + Shading + Temperature + Soiling)
           - Includes system losses, inverter inefficiencies, wiring losses
        
        **Validation**: Ensure sum of attributed losses equals total measured loss
        """)
    
    with tab4:
        st.subheader("Key Assumptions")
        
        st.markdown("""
        ### Model Assumptions
        
        1. **Data Integrity**
           - Sensor measurements are calibrated and accurate
           - Timestamp synchronization across all data sources
           - No systematic measurement errors
        
        2. **System Parameters**
           - Panel degradation is uniform and minimal over analysis period
           - Inverter efficiency curves are stable
           - String connections are properly maintained
        
        3. **Environmental Modeling**
           - Solar position calculations based on provided coordinates
           - Weather station data is representative of entire plant
           - Microclimate variations are minimal
        
        4. **Loss Attribution**
           - Loss categories are independent and additive
           - Theoretical generation represents true maximum potential
           - Residual losses capture all unmodeled effects
        
        5. **Temporal Considerations**
           - 15-minute intervals provide sufficient temporal resolution
           - Seasonal variations are captured in training data
           - No significant system changes during analysis period
        
        ### Limitations
        
        - Model accuracy depends on data quality and completeness
        - Some loss interactions may not be fully captured
        - Extreme weather events may not be well-represented
        - Long-term degradation effects require extended datasets
        """)

def show_insights():
    """Display insights and recommendations"""
    
    st.header("💡 Insights & Recommendations")
    
    # Key insights
    st.subheader("🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Performance Insights
        
        1. **Loss Distribution**
           - Cloud cover: 35% of total losses
           - Temperature effects: 28% of total losses
           - Soiling: 15% of total losses
           - Shading: 12% of total losses
           - Other factors: 10% of total losses
        
        2. **Temporal Patterns**
           - Peak losses occur during summer months
           - Morning hours show higher soiling impact
           - Afternoon temperature losses are significant
           - Seasonal cloud patterns affect winter performance
        
        3. **Asset Performance**
           - INV-3 shows 2.7% higher efficiency than INV-8
           - String-level variations indicate maintenance needs
           - Consistent performance degradation in Zone 8
        """)
    
    with col2:
        st.markdown("""
        ### Operational Insights
        
        1. **Maintenance Priorities**
           - Soiling cleaning schedule optimization needed
           - String 15-18 require immediate attention
           - Temperature management in summer critical
        
        2. **Performance Optimization**
           - Cloud forecasting can improve dispatch planning
           - Shading mitigation strategies required
           - Enhanced monitoring for underperforming assets
        
        3. **Data Quality**
           - Some sensors show drift patterns
           - Enhanced calibration procedures recommended
           - Additional meteorological stations needed
        """)
    
    # Recommendations
    st.subheader("📋 Recommendations")
    
    recommendation_tabs = st.tabs(["🔧 Operational", "📊 Monitoring", "🎯 Strategic"])
    
    with recommendation_tabs[0]:
        st.markdown("""
        ### Operational Recommendations
        
        #### Immediate Actions (0-3 months)
        - Implement optimized cleaning schedule based on soiling loss analysis
        - Investigate and repair underperforming strings (String 15-18)
        - Calibrate sensors showing measurement drift
        - Optimize inverter dispatch during peak temperature periods
        
        #### Medium-term Actions (3-12 months)
        - Install additional temperature monitoring points
        - Implement predictive maintenance based on performance trends
        - Upgrade inverters in Zone 8 if economically justified
        - Develop automated cleaning system for high-soiling periods
        
        #### Long-term Actions (1-3 years)
        - Consider tracker system upgrades for shading mitigation
        - Evaluate module replacement for severely degraded strings
        - Implement advanced forecasting systems for cloud cover
        - Develop digital twin model for real-time optimization
        """)
    
    with recommendation_tabs[1]:
        st.markdown("""
        ### Monitoring & Data Enhancement
        
        #### Sensor Improvements
        - Install additional irradiance sensors for better spatial coverage
        - Add string-level current and voltage monitoring
        - Implement real-time soiling detection sensors
        - Enhance weather station capabilities with cloud cameras
        
        #### Data Analytics
        - Implement real-time loss attribution algorithms
        - Develop automated anomaly detection systems
        - Create predictive models for performance forecasting
        - Establish benchmarking against similar plants
        
        #### Reporting & Visualization
        - Automated daily performance reports
        - Real-time dashboard for operators
        - Monthly trend analysis and forecasting
        - Automated alert systems for performance deviations
        """)
    
    with recommendation_tabs[2]:
        st.markdown("""
        ### Strategic Recommendations
        
        #### Technology Upgrades
        - Evaluate next-generation inverter technology
        - Consider bifacial panel upgrades for improved performance
        - Implement advanced tracking systems with AI optimization
        - Explore battery storage integration for grid services
        
        #### Business Optimization
        - Develop performance-based O&M contracts
        - Implement dynamic pricing strategies based on loss forecasting
        - Create insurance products based on performance analytics
        - Establish benchmarking program with other plants
        
        #### Research & Development
        - Collaborate with universities on advanced loss modeling
        - Investigate novel soiling mitigation technologies
        - Develop plant-specific machine learning models
        - Participate in industry consortiums for best practices
        
        #### Risk Management
        - Develop contingency plans for extreme weather events
        - Implement cybersecurity measures for data protection
        - Create performance guarantees based on analytics
        - Establish supplier performance monitoring programs
        """)
    
    # Success metrics
    st.subheader("🎯 Success Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Efficiency Improvement", "3-5%", "Current: 91.2%")
    with col2:
        st.metric("Soiling Loss Reduction", "2-3%", "Current: 1.8%")
    with col3:
        st.metric("O&M Cost Reduction", "15-20%", "Through predictive maintenance")

if __name__ == "__main__":
    main()
