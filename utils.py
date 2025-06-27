import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import os
import logging
from scipy import stats

class Utils:
    """
    Utility class with helper functions for solar energy analysis
    """
    
    def __init__(self):
        """Initialize utilities"""
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def validate_data_quality(data):
        """
        Validate data quality and return quality metrics
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Data quality metrics
        """
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['total_records'] = len(data)
        quality_metrics['total_columns'] = len(data.columns)
        
        # Missing values analysis
        missing_values = data.isnull().sum()
        quality_metrics['missing_values'] = missing_values.to_dict()
        quality_metrics['columns_with_missing'] = (missing_values > 0).sum()
        quality_metrics['missing_percentage'] = (missing_values.sum() / (len(data) * len(data.columns)) * 100)
        
        # Data types
        quality_metrics['data_types'] = data.dtypes.to_dict()
        quality_metrics['numeric_columns'] = len(data.select_dtypes(include=[np.number]).columns)
        quality_metrics['categorical_columns'] = len(data.select_dtypes(include=['object']).columns)
        
        # Duplicates
        quality_metrics['duplicate_rows'] = data.duplicated().sum()
        
        # Date range (if datetime column exists)
        if 'datetime' in data.columns:
            quality_metrics['date_range'] = {
                'start': data['datetime'].min(),
                'end': data['datetime'].max(),
                'duration_days': (data['datetime'].max() - data['datetime'].min()).days
            }
        
        return quality_metrics
    
    @staticmethod
    def calculate_system_capacity():
        """
        Calculate system capacity based on plant specifications
        
        Returns:
            dict: System capacity information
        """
        return {
            'total_capacity_mw': 7.6,  # 2 x 3.8 MW inverters
            'inv_03_capacity_mw': 3.8,
            'inv_08_capacity_mw': 3.8,
            'plant_capacity_mwh': 45.6,  # From PDF
            'location': {
                'latitude': 38.0022,  # 38° 0' 2" N
                'longitude': -1.3344,  # 1° 20' 4" W
                'timezone': 'UTC+1'
            }
        }
    
    @staticmethod
    def format_energy_value(value, unit='kWh'):
        """
        Format energy values for display
        
        Args:
            value (float): Energy value
            unit (str): Energy unit
            
        Returns:
            str: Formatted energy value
        """
        if pd.isna(value):
            return "N/A"
        
        if abs(value) >= 1000000:
            return f"{value/1000000:.2f} G{unit}"
        elif abs(value) >= 1000:
            return f"{value/1000:.2f} M{unit}"
        else:
            return f"{value:.2f} {unit}"
    
    @staticmethod
    def format_percentage(value):
        """
        Format percentage values for display
        
        Args:
            value (float): Percentage value (0-100)
            
        Returns:
            str: Formatted percentage
        """
        if pd.isna(value):
            return "N/A"
        return f"{value:.2f}%"
    
    @staticmethod
    def calculate_performance_metrics(actual, theoretical):
        """
        Calculate performance metrics
        
        Args:
            actual (pd.Series): Actual generation values
            theoretical (pd.Series): Theoretical generation values
            
        Returns:
            dict: Performance metrics
        """
        if len(actual) == 0 or len(theoretical) == 0:
            return {
                'efficiency': 0,
                'total_losses': 0,
                'average_losses': 0,
                'performance_ratio': 0
            }
        
        # Ensure both series have the same length
        min_length = min(len(actual), len(theoretical))
        actual = actual.iloc[:min_length]
        theoretical = theoretical.iloc[:min_length]
        
        # Calculate metrics
        efficiency = (actual.sum() / theoretical.sum() * 100) if theoretical.sum() > 0 else 0
        total_losses = theoretical.sum() - actual.sum()
        average_losses = total_losses / len(actual) if len(actual) > 0 else 0
        performance_ratio = actual.mean() / theoretical.mean() if theoretical.mean() > 0 else 0
        
        return {
            'efficiency': efficiency,
            'total_losses': total_losses,
            'average_losses': average_losses,
            'performance_ratio': performance_ratio
        }
    
    @staticmethod
    def create_time_filters(data):
        """
        Create time-based filter options
        
        Args:
            data (pd.DataFrame): Data with datetime column
            
        Returns:
            dict: Time filter options
        """
        if 'datetime' not in data.columns:
            return {}
        
        return {
            'date_range': {
                'min': data['datetime'].min().date(),
                'max': data['datetime'].max().date()
            },
            'hours': sorted(data['datetime'].dt.hour.unique()),
            'months': sorted(data['datetime'].dt.month.unique()),
            'years': sorted(data['datetime'].dt.year.unique())
        }
    
    @staticmethod
    def aggregate_by_period(data, period='H'):
        """
        Aggregate data by time period
        
        Args:
            data (pd.DataFrame): Input data with datetime index
            period (str): Aggregation period ('H', 'D', 'W', 'M')
            
        Returns:
            pd.DataFrame: Aggregated data
        """
        if 'datetime' not in data.columns:
            return data
        
        # Set datetime as index if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime')
        
        # Select numeric columns for aggregation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Aggregate based on column type
        agg_functions = {}
        for col in numeric_columns:
            if any(keyword in col.lower() for keyword in ['power', 'energy', 'current', 'voltage']):
                agg_functions[col] = 'sum'  # Sum for energy/power values
            else:
                agg_functions[col] = 'mean'  # Mean for other measurements
        
        # Perform aggregation
        aggregated = data.groupby(pd.Grouper(freq=period)).agg(agg_functions)
        
        return aggregated.reset_index()
    
    @staticmethod
    def detect_outliers(data, column, method='iqr', threshold=1.5):
        """
        Detect outliers in a data column
        
        Args:
            data (pd.DataFrame): Input data
            column (str): Column name to analyze
            method (str): Outlier detection method ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        if column not in data.columns:
            return pd.Series([False] * len(data))
        
        values = data[column]
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            outliers = z_scores > threshold
        
        else:
            outliers = pd.Series([False] * len(data))
        
        return outliers
    
    @staticmethod
    def calculate_solar_position_simple(datetime_series, latitude=38.0022, longitude=-1.3344):
        """
        Calculate simplified solar position (elevation angle)
        
        Args:
            datetime_series (pd.Series): Datetime series
            latitude (float): Latitude in degrees
            longitude (float): Longitude in degrees
            
        Returns:
            pd.DataFrame: Solar position data
        """
        # Convert to numpy for vectorized operations
        dates = pd.to_datetime(datetime_series)
        
        # Day of year
        day_of_year = dates.dayofyear
        
        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (dates.hour + dates.minute/60 - 12)
        
        # Solar elevation angle
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        elevation = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(decl_rad) + 
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad)
        ))
        
        # Solar azimuth angle
        azimuth = np.degrees(np.arctan2(
            np.sin(hour_rad),
            np.cos(hour_rad) * np.sin(lat_rad) - np.tan(decl_rad) * np.cos(lat_rad)
        ))
        
        return pd.DataFrame({
            'solar_elevation': elevation,
            'solar_azimuth': azimuth,
            'solar_declination': declination
        })
    
    @staticmethod
    def export_results_to_csv(data, filename, include_timestamp=True):
        """
        Export results to CSV file
        
        Args:
            data (pd.DataFrame): Data to export
            filename (str): Output filename
            include_timestamp (bool): Include timestamp in filename
            
        Returns:
            str: Full filepath of exported file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join("exports", filename)
        
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        # Export data
        data.to_csv(filepath, index=False)
        
        return filepath
    
    @staticmethod
    def create_summary_statistics(data, groupby_column=None):
        """
        Create summary statistics for numerical columns
        
        Args:
            data (pd.DataFrame): Input data
            groupby_column (str): Column to group by (optional)
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if groupby_column and groupby_column in data.columns:
            summary = numeric_data.groupby(data[groupby_column]).agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(3)
        else:
            summary = numeric_data.agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(3)
        
        return summary
    
    @staticmethod
    def validate_datetime_series(datetime_series):
        """
        Validate datetime series for consistency and gaps
        
        Args:
            datetime_series (pd.Series): Datetime series
            
        Returns:
            dict: Validation results
        """
        if len(datetime_series) == 0:
            return {'valid': False, 'message': 'Empty datetime series'}
        
        # Check for duplicates
        duplicates = datetime_series.duplicated().sum()
        
        # Check for proper sorting
        is_sorted = datetime_series.is_monotonic_increasing
        
        # Check for gaps (assuming 15-minute intervals)
        if len(datetime_series) > 1:
            time_diffs = datetime_series.diff().dropna()
            expected_interval = pd.Timedelta(minutes=15)
            gaps = (time_diffs > expected_interval * 1.5).sum()  # Allow 50% tolerance
        else:
            gaps = 0
        
        return {
            'valid': duplicates == 0 and is_sorted and gaps < len(datetime_series) * 0.05,
            'duplicates': duplicates,
            'is_sorted': is_sorted,
            'gaps': gaps,
            'message': f'Duplicates: {duplicates}, Sorted: {is_sorted}, Gaps: {gaps}'
        }
    
    @staticmethod
    def create_streamlit_metrics(data_dict, cols=4):
        """
        Create Streamlit metrics display
        
        Args:
            data_dict (dict): Dictionary of metric name and value pairs
            cols (int): Number of columns for metric display
        """
        if not data_dict:
            st.warning("No metrics to display")
            return
        
        # Create columns
        metric_cols = st.columns(cols)
        
        # Display metrics
        for i, (key, value) in enumerate(data_dict.items()):
            col_idx = i % cols
            
            # Format value based on type
            if isinstance(value, (int, float)):
                if abs(value) >= 1000000:
                    display_value = f"{value/1000000:.2f}M"
                elif abs(value) >= 1000:
                    display_value = f"{value/1000:.2f}K"
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            with metric_cols[col_idx]:
                st.metric(key.replace('_', ' ').title(), display_value)
    
    @staticmethod
    def safe_divide(numerator, denominator, default=0):
        """
        Safely divide two values, handling division by zero
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division by zero
            
        Returns:
            float: Division result or default value
        """
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return default
        return numerator / denominator
    
    @staticmethod
    def format_duration(seconds):
        """
        Format duration in seconds to human-readable format
        
        Args:
            seconds (float): Duration in seconds
            
        Returns:
            str: Formatted duration
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"
    
    def log_info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def log_error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
