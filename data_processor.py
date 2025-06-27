import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing class for solar PV plant data
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        
    def load_data(self, file_path):
        """
        Load solar PV plant data from CSV file
        
        Args:
            file_path (str): Path to the CSV data file
            
        Returns:
            pd.DataFrame: Loaded raw data
        """
        try:
            # Load data with proper datetime parsing
            data = pd.read_csv(file_path)
            
            # Convert datetime column
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self, data):
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Processed data
        """
        print("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # 1. Handle datetime features
        processed_data = self._process_datetime_features(processed_data)
        
        # 2. Feature engineering
        processed_data = self._engineer_features(processed_data)
        
        # 3. Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # 4. Outlier detection and treatment
        processed_data = self._handle_outliers(processed_data)
        
        # 5. Data validation
        processed_data = self._validate_data(processed_data)
        
        print(f"Data preprocessing completed. Shape: {processed_data.shape}")
        
        return processed_data
    
    def _process_datetime_features(self, data):
        """
        Process datetime column and extract temporal features
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with datetime features
        """
        if 'datetime' not in data.columns:
            return data
        
        # Extract temporal features
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['day'] = data['datetime'].dt.day
        data['hour'] = data['datetime'].dt.hour
        data['minute'] = data['datetime'].dt.minute
        data['day_of_year'] = data['datetime'].dt.dayofyear
        data['day_of_week'] = data['datetime'].dt.dayofweek
        
        # Season encoding
        data['season'] = data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Solar position calculations (simplified)
        data = self._calculate_solar_position(data)
        
        return data
    
    def _calculate_solar_position(self, data):
        """
        Calculate approximate solar position angles
        
        Args:
            data (pd.DataFrame): Input data with datetime
            
        Returns:
            pd.DataFrame: Data with solar position features
        """
        # Plant coordinates (from PDF)
        latitude = 38.0022  # 38° 0' 2" N
        longitude = -1.3344  # 1° 20' 4" W
        
        # Simplified solar position calculation
        # This is an approximation - for production use, consider pyephem or pvlib
        
        # Day angle
        data['day_angle'] = 2 * np.pi * data['day_of_year'] / 365.25
        
        # Solar declination (simplified)
        data['solar_declination'] = 23.45 * np.sin(np.radians(360 * (284 + data['day_of_year']) / 365))
        
        # Hour angle
        data['hour_angle'] = 15 * (data['hour'] + data['minute']/60 - 12)
        
        # Solar elevation angle (simplified)
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(data['solar_declination'])
        hour_rad = np.radians(data['hour_angle'])
        
        data['solar_elevation'] = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(decl_rad) + 
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad)
        ))
        
        # Solar azimuth angle (simplified)
        data['solar_azimuth'] = np.degrees(np.arctan2(
            np.sin(hour_rad),
            np.cos(hour_rad) * np.sin(lat_rad) - np.tan(decl_rad) * np.cos(lat_rad)
        ))
        
        # Air mass (simplified)
        data['air_mass'] = np.where(
            data['solar_elevation'] > 0,
            1 / np.sin(np.radians(data['solar_elevation'])),
            np.nan
        )
        
        return data
    
    def _engineer_features(self, data):
        """
        Engineer relevant features for solar PV analysis
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # Extract system identifiers
        data = self._extract_system_identifiers(data)
        
        # Calculate derived meteorological features
        data = self._calculate_meteorological_features(data)
        
        # Calculate electrical performance features
        data = self._calculate_electrical_features(data)
        
        # Create interaction features
        data = self._create_interaction_features(data)
        
        return data
    
    def _extract_system_identifiers(self, data):
        """
        Extract system identifiers (inverters, strings, zones) from column names
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with system identifiers
        """
        # Initialize system identifier columns
        data['zone'] = 'Unknown'
        data['inverter'] = 'Unknown'
        data['string'] = 'Unknown'
        
        # Extract zone information from column names
        for col in data.columns:
            if 'ctin03' in col.lower():
                data['zone'] = 'CTIN03'
                data['inverter'] = 'INV-3'
            elif 'ctin08' in col.lower():
                data['zone'] = 'CTIN08'
                data['inverter'] = 'INV-8'
        
        # Extract string information from column names
        string_columns = [col for col in data.columns if 'string' in col.lower()]
        if string_columns:
            # For demonstration, assign strings based on available data
            data['string'] = 'String_1'  # This would be extracted from actual data structure
        
        return data
    
    def _calculate_meteorological_features(self, data):
        """
        Calculate derived meteorological features
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with meteorological features
        """
        # Temperature-related features
        temp_columns = [col for col in data.columns if 't_amb' in col]
        if temp_columns:
            data['avg_ambient_temp'] = data[temp_columns].mean(axis=1)
            data['temp_variance'] = data[temp_columns].var(axis=1)
        
        # Irradiance-related features
        gii_columns = [col for col in data.columns if 'gii' in col and 'rear' not in col]
        if gii_columns:
            data['avg_gii'] = data[gii_columns].mean(axis=1)
            data['gii_variance'] = data[gii_columns].var(axis=1)
        
        ghi_columns = [col for col in data.columns if 'ghi' in col]
        if ghi_columns:
            data['avg_ghi'] = data[ghi_columns].mean(axis=1)
        
        # Wind-related features
        wind_columns = [col for col in data.columns if 'ws' in col or 'wind' in col]
        if wind_columns:
            data['avg_wind_speed'] = data[wind_columns].mean(axis=1)
        
        # Humidity features
        humidity_columns = [col for col in data.columns if 'h_r' in col]
        if humidity_columns:
            data['avg_humidity'] = data[humidity_columns].mean(axis=1)
        
        # Clear sky index (GHI/theoretical clear sky GHI)
        if 'avg_ghi' in data.columns and 'solar_elevation' in data.columns:
            # Simplified clear sky model
            data['clear_sky_ghi'] = np.where(
                data['solar_elevation'] > 0,
                1361 * np.sin(np.radians(data['solar_elevation'])) * 0.7,  # Simplified
                0
            )
            data['clear_sky_index'] = np.where(
                data['clear_sky_ghi'] > 0,
                data['avg_ghi'] / data['clear_sky_ghi'],
                0
            )
        
        return data
    
    def _calculate_electrical_features(self, data):
        """
        Calculate electrical performance features
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with electrical features
        """
        # DC power features
        dc_power_columns = [col for col in data.columns if 'p_dc' in col]
        if dc_power_columns:
            data['total_dc_power'] = data[dc_power_columns].sum(axis=1)
        
        # AC power features
        ac_power_columns = [col for col in data.columns if 'p' in col and 'p_dc' not in col]
        ac_power_columns = [col for col in ac_power_columns if any(x in col for x in ['inv', 'ppc'])]
        if ac_power_columns:
            data['total_ac_power'] = data[ac_power_columns].sum(axis=1)
        
        # Current features
        current_columns = [col for col in data.columns if 'pv_i' in col]
        if current_columns:
            data['total_current'] = data[current_columns].sum(axis=1)
            data['avg_current'] = data[current_columns].mean(axis=1)
        
        # Voltage features
        voltage_columns = [col for col in data.columns if 'pv_v' in col]
        if voltage_columns:
            data['avg_voltage'] = data[voltage_columns].mean(axis=1)
            data['voltage_variance'] = data[voltage_columns].var(axis=1)
        
        # Efficiency calculation (if both DC and AC power available)
        if 'total_dc_power' in data.columns and 'total_ac_power' in data.columns:
            data['inverter_efficiency'] = np.where(
                data['total_dc_power'] > 0,
                data['total_ac_power'] / data['total_dc_power'],
                0
            )
        
        return data
    
    def _create_interaction_features(self, data):
        """
        Create interaction features between different variables
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with interaction features
        """
        # Temperature-irradiance interactions
        if 'avg_ambient_temp' in data.columns and 'avg_gii' in data.columns:
            data['temp_irradiance_interaction'] = data['avg_ambient_temp'] * data['avg_gii']
        
        # Wind-temperature interactions
        if 'avg_wind_speed' in data.columns and 'avg_ambient_temp' in data.columns:
            data['wind_temp_interaction'] = data['avg_wind_speed'] * data['avg_ambient_temp']
        
        # Humidity-temperature interactions
        if 'avg_humidity' in data.columns and 'avg_ambient_temp' in data.columns:
            data['humidity_temp_interaction'] = data['avg_humidity'] * data['avg_ambient_temp']
        
        # Solar position-irradiance interactions
        if 'solar_elevation' in data.columns and 'avg_gii' in data.columns:
            data['elevation_irradiance_interaction'] = data['solar_elevation'] * data['avg_gii']
        
        return data
    
    def _handle_missing_values(self, data):
        """
        Handle missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        # Separate numerical and categorical columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        for col in numerical_columns:
            if data[col].isnull().sum() > 0:
                # Use forward fill for time series data, then backward fill
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use median
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(data[col].median())
        
        # Handle categorical missing values
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
        
        return data
    
    def _handle_outliers(self, data):
        """
        Detect and handle outliers in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with handled outliers
        """
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if col not in ['year', 'month', 'day', 'hour', 'minute']:  # Skip datetime components
                # Calculate IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        return data
    
    def _validate_data(self, data):
        """
        Validate the processed data
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Validated data
        """
        # Check for infinite values
        inf_columns = data.columns[data.isin([np.inf, -np.inf]).any()].tolist()
        if inf_columns:
            print(f"Warning: Infinite values found in columns: {inf_columns}")
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
        
        # Check for remaining missing values
        missing_columns = data.columns[data.isnull().any()].tolist()
        if missing_columns:
            print(f"Warning: Missing values still present in columns: {missing_columns}")
        
        # Ensure datetime is properly formatted
        if 'datetime' in data.columns:
            data = data.sort_values('datetime').reset_index(drop=True)
        
        return data
    
    def prepare_ml_features(self, data, target_column=None):
        """
        Prepare features for machine learning
        
        Args:
            data (pd.DataFrame): Processed data
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y) features and target
        """
        # Identify feature columns (exclude datetime and system identifiers)
        exclude_columns = ['datetime', 'zone', 'inverter', 'string', 'season']
        if target_column:
            exclude_columns.append(target_column)
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        feature_columns = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare features
        X = data[feature_columns].copy()
        
        # Handle categorical variables if any remain
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Prepare target
        y = None
        if target_column and target_column in data.columns:
            y = data[target_column].copy()
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def get_feature_names(self):
        """
        Get list of feature names
        
        Returns:
            list: Feature column names
        """
        return self.feature_columns
