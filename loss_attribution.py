import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LossAttributor:
    """
    Advanced loss attribution class for solar PV energy losses
    """
    
    def __init__(self):
        """Initialize the loss attributor"""
        self.loss_models = {}
        self.loss_coefficients = {}
        self.attribution_results = {}
        
    def attribute_losses(self, data, aggregation_level='Plant', time_granularity='15-minute'):
        """
        Perform comprehensive loss attribution analysis
        
        Args:
            data (pd.DataFrame): Processed solar data
            aggregation_level (str): Level of analysis ('Plant', 'Inverter', 'String')
            time_granularity (str): Time granularity for analysis
            
        Returns:
            dict: Loss attribution results
        """
        print(f"Starting loss attribution analysis at {aggregation_level} level...")
        
        # Prepare data for analysis
        analysis_data = self._prepare_analysis_data(data, aggregation_level, time_granularity)
        
        # Calculate theoretical generation
        theoretical_generation = self._calculate_theoretical_generation(analysis_data)
        
        # Calculate actual generation
        actual_generation = self._calculate_actual_generation(analysis_data)
        
        # Calculate total losses
        total_losses = theoretical_generation - actual_generation
        
        # Attribute losses to specific causes
        loss_attribution = self._attribute_specific_losses(
            analysis_data, theoretical_generation, actual_generation, total_losses
        )
        
        # Aggregate results
        results = self._aggregate_results(
            analysis_data, loss_attribution, time_granularity
        )
        
        print("Loss attribution analysis completed!")
        return results
    
    def _prepare_analysis_data(self, data, aggregation_level, time_granularity):
        """
        Prepare data for loss attribution analysis
        
        Args:
            data (pd.DataFrame): Input data
            aggregation_level (str): Aggregation level
            time_granularity (str): Time granularity
            
        Returns:
            pd.DataFrame: Prepared analysis data
        """
        analysis_data = data.copy()
        
        # Aggregate by time granularity
        if time_granularity != '15-minute':
            analysis_data = self._aggregate_by_time(analysis_data, time_granularity)
        
        # Aggregate by system level
        if aggregation_level != 'String':
            analysis_data = self._aggregate_by_level(analysis_data, aggregation_level)
        
        return analysis_data
    
    def _aggregate_by_time(self, data, time_granularity):
        """
        Aggregate data by specified time granularity
        
        Args:
            data (pd.DataFrame): Input data
            time_granularity (str): Time granularity
            
        Returns:
            pd.DataFrame: Time-aggregated data
        """
        if 'datetime' not in data.columns:
            return data
        
        # Define aggregation frequency
        freq_map = {
            'Hourly': 'H',
            'Daily': 'D',
            'Weekly': 'W',
            'Monthly': 'M'
        }
        
        freq = freq_map.get(time_granularity, 'H')
        
        # Group by time period
        data_grouped = data.set_index('datetime').groupby(pd.Grouper(freq=freq))
        
        # Aggregate numerical columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        aggregated_data = data_grouped[numerical_columns].agg({
            col: 'mean' if col in ['temp', 'humidity', 'efficiency'] else 'sum'
            for col in numerical_columns
        }).reset_index()
        
        return aggregated_data
    
    def _aggregate_by_level(self, data, aggregation_level):
        """
        Aggregate data by system level
        
        Args:
            data (pd.DataFrame): Input data
            aggregation_level (str): System level
            
        Returns:
            pd.DataFrame: Level-aggregated data
        """
        if aggregation_level == 'Plant':
            # Aggregate all data to plant level
            numerical_columns = data.select_dtypes(include=[np.number]).columns
            aggregated = data.groupby(['datetime'] if 'datetime' in data.columns else [data.index])[numerical_columns].sum().reset_index()
            aggregated['level'] = 'Plant'
            return aggregated
        
        elif aggregation_level == 'Inverter':
            # Aggregate by inverter
            if 'inverter' in data.columns:
                numerical_columns = data.select_dtypes(include=[np.number]).columns
                group_cols = ['datetime', 'inverter'] if 'datetime' in data.columns else ['inverter']
                aggregated = data.groupby(group_cols)[numerical_columns].sum().reset_index()
                return aggregated
        
        return data
    
    def _calculate_theoretical_generation(self, data):
        """
        Calculate theoretical maximum generation
        
        Args:
            data (pd.DataFrame): Analysis data
            
        Returns:
            pd.Series: Theoretical generation values
        """
        # Use advanced modeling to estimate theoretical generation
        # This would typically use clear-sky irradiance models and system specifications
        
        # For demonstration, using a combination of available irradiance data
        irradiance_columns = [col for col in data.columns if 'gii' in col.lower() or 'ghi' in col.lower()]
        
        if irradiance_columns:
            # Base theoretical generation on irradiance
            avg_irradiance = data[irradiance_columns].mean(axis=1)
            
            # System specifications (from PDF)
            system_capacity = 7.6  # MW (2 x 3.8 MW inverters)
            module_efficiency = 0.20  # 20% efficiency
            system_efficiency = 0.85  # 85% system efficiency
            
            # Calculate theoretical generation (simplified model)
            theoretical_gen = (avg_irradiance / 1000) * system_capacity * module_efficiency * system_efficiency
            
            # Add temperature corrections for ideal conditions
            if 'avg_ambient_temp' in data.columns:
                # Ideal temperature is 25°C
                temp_correction = 1 + 0.004 * (25 - data['avg_ambient_temp'])  # 0.4%/°C temperature coefficient
                theoretical_gen = theoretical_gen * temp_correction
            
            return theoretical_gen
        else:
            # Fallback: use power data if available
            power_columns = [col for col in data.columns if 'p' in col.lower() and data[col].dtype in ['int64', 'float64']]
            if power_columns:
                return data[power_columns].sum(axis=1) * 1.2  # Assume 20% losses in actual data
            else:
                return pd.Series([1.0] * len(data))  # Default values
    
    def _calculate_actual_generation(self, data):
        """
        Calculate actual generation from available data
        
        Args:
            data (pd.DataFrame): Analysis data
            
        Returns:
            pd.Series: Actual generation values
        """
        # Look for actual generation columns
        actual_gen_columns = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['eact', 'export', 'p_tot']):
                if data[col].dtype in ['int64', 'float64']:
                    actual_gen_columns.append(col)
        
        if actual_gen_columns:
            return data[actual_gen_columns].sum(axis=1)
        else:
            # Use power columns as proxy
            power_columns = [col for col in data.columns if 'p' in col.lower() and data[col].dtype in ['int64', 'float64']]
            if power_columns:
                return data[power_columns].sum(axis=1)
            else:
                return pd.Series([0.8] * len(data))  # Default values
    
    def _attribute_specific_losses(self, data, theoretical_gen, actual_gen, total_losses):
        """
        Attribute losses to specific causes using advanced statistical methods
        
        Args:
            data (pd.DataFrame): Analysis data
            theoretical_gen (pd.Series): Theoretical generation
            actual_gen (pd.Series): Actual generation
            total_losses (pd.Series): Total losses
            
        Returns:
            dict: Attribution of losses to specific causes
        """
        loss_attribution = {}
        
        # 1. Cloud Cover Losses
        loss_attribution['cloud_losses'] = self._calculate_cloud_losses(
            data, theoretical_gen, actual_gen
        )
        
        # 2. Shading Losses
        loss_attribution['shading_losses'] = self._calculate_shading_losses(
            data, theoretical_gen, actual_gen
        )
        
        # 3. Temperature Losses
        loss_attribution['temperature_losses'] = self._calculate_temperature_losses(
            data, theoretical_gen, actual_gen
        )
        
        # 4. Soiling Losses
        loss_attribution['soiling_losses'] = self._calculate_soiling_losses(
            data, theoretical_gen, actual_gen
        )
        
        # 5. Other/Residual Losses
        attributed_losses = (
            loss_attribution['cloud_losses'] + 
            loss_attribution['shading_losses'] + 
            loss_attribution['temperature_losses'] + 
            loss_attribution['soiling_losses']
        )
        
        loss_attribution['other_losses'] = total_losses - attributed_losses
        
        # Ensure other losses are non-negative
        loss_attribution['other_losses'] = np.maximum(loss_attribution['other_losses'], 0)
        
        return loss_attribution
    
    def _calculate_cloud_losses(self, data, theoretical_gen, actual_gen):
        """
        Calculate energy losses due to cloud cover
        
        Args:
            data (pd.DataFrame): Analysis data
            theoretical_gen (pd.Series): Theoretical generation
            actual_gen (pd.Series): Actual generation
            
        Returns:
            pd.Series: Cloud cover losses
        """
        # Method 1: Clear sky index approach
        if 'clear_sky_index' in data.columns:
            # Cloud losses based on clear sky index
            cloud_factor = 1 - data['clear_sky_index']
            cloud_losses = theoretical_gen * cloud_factor * 0.8  # 80% of theoretical reduction
            return np.maximum(cloud_losses, 0)
        
        # Method 2: Irradiance variability approach
        irradiance_columns = [col for col in data.columns if 'gii' in col.lower() or 'ghi' in col.lower()]
        if irradiance_columns:
            avg_irradiance = data[irradiance_columns].mean(axis=1)
            irradiance_variance = data[irradiance_columns].var(axis=1)
            
            # High variance indicates clouds
            cloud_indicator = irradiance_variance / (avg_irradiance + 1e-6)
            cloud_losses = theoretical_gen * cloud_indicator * 0.001  # Scale factor
            return np.maximum(cloud_losses, 0)
        
        # Method 3: Statistical model based on generation patterns
        # Identify periods with rapid power fluctuations (cloud effects)
        if len(actual_gen) > 1:
            power_diff = np.abs(actual_gen.diff())
            cloud_indicator = power_diff / (actual_gen + 1e-6)
            cloud_losses = theoretical_gen * cloud_indicator * 0.1
            return np.maximum(cloud_losses, 0)
        
        return pd.Series([0.0] * len(data))
    
    def _calculate_shading_losses(self, data, theoretical_gen, actual_gen):
        """
        Calculate energy losses due to shading
        
        Args:
            data (pd.DataFrame): Analysis data
            theoretical_gen (pd.Series): Theoretical generation
            actual_gen (pd.Series): Actual generation
            
        Returns:
            pd.Series: Shading losses
        """
        # Method 1: Solar elevation based shading
        if 'solar_elevation' in data.columns:
            # Shading is more likely at low solar elevations
            shading_factor = np.exp(-data['solar_elevation'] / 20)  # Exponential decay
            shading_losses = theoretical_gen * shading_factor * 0.1
            return np.maximum(shading_losses, 0)
        
        # Method 2: Time-based shading patterns
        if 'hour' in data.columns:
            # Morning and evening shading
            morning_shading = np.exp(-(data['hour'] - 6)**2 / 8)  # Peak at 6 AM
            evening_shading = np.exp(-(data['hour'] - 18)**2 / 8)  # Peak at 6 PM
            shading_factor = (morning_shading + evening_shading) * 0.1
            shading_losses = theoretical_gen * shading_factor
            return np.maximum(shading_losses, 0)
        
        # Method 3: Irradiance deviation analysis
        irradiance_columns = [col for col in data.columns if 'desviacion' in col.lower() or 'deviation' in col.lower()]
        if irradiance_columns:
            avg_deviation = data[irradiance_columns].mean(axis=1)
            shading_losses = theoretical_gen * np.abs(avg_deviation) * 0.0001
            return np.maximum(shading_losses, 0)
        
        return pd.Series([0.0] * len(data))
    
    def _calculate_temperature_losses(self, data, theoretical_gen, actual_gen):
        """
        Calculate energy losses due to high temperatures
        
        Args:
            data (pd.DataFrame): Analysis data
            theoretical_gen (pd.Series): Theoretical generation
            actual_gen (pd.Series): Actual generation
            
        Returns:
            pd.Series: Temperature losses
        """
        # Method 1: Direct temperature coefficient model
        if 'avg_ambient_temp' in data.columns:
            # Standard temperature coefficient for silicon PV: -0.4%/°C
            temp_coeff = -0.004  # per °C
            reference_temp = 25  # °C (STC)
            
            # Calculate cell temperature (simplified)
            # Cell temp ≈ Ambient temp + (Irradiance / 1000) * 25
            irradiance_columns = [col for col in data.columns if 'gii' in col.lower()]
            if irradiance_columns:
                avg_irradiance = data[irradiance_columns].mean(axis=1)
                cell_temp = data['avg_ambient_temp'] + (avg_irradiance / 1000) * 25
            else:
                cell_temp = data['avg_ambient_temp'] + 10  # Approximate difference
            
            # Temperature losses
            temp_losses = theoretical_gen * temp_coeff * (cell_temp - reference_temp)
            return np.maximum(temp_losses, 0)
        
        # Method 2: Module temperature approach
        module_temp_columns = [col for col in data.columns if 't_mod' in col.lower()]
        if module_temp_columns:
            avg_module_temp = data[module_temp_columns].mean(axis=1)
            temp_losses = theoretical_gen * -0.004 * (avg_module_temp - 25)
            return np.maximum(temp_losses, 0)
        
        # Method 3: Hour-based temperature model
        if 'hour' in data.columns:
            # Assume temperature peaks at 2 PM
            temp_factor = np.exp(-((data['hour'] - 14)**2) / 12)  # Peak at 2 PM
            temp_losses = theoretical_gen * temp_factor * 0.15
            return np.maximum(temp_losses, 0)
        
        return pd.Series([0.0] * len(data))
    
    def _calculate_soiling_losses(self, data, theoretical_gen, actual_gen):
        """
        Calculate energy losses due to soiling
        
        Args:
            data (pd.DataFrame): Analysis data
            theoretical_gen (pd.Series): Theoretical generation
            actual_gen (pd.Series): Actual generation
            
        Returns:
            pd.Series: Soiling losses
        """
        # Method 1: Reference cell approach
        clean_cell_columns = [col for col in data.columns if 'cel_1' in col.lower()]
        dirty_cell_columns = [col for col in data.columns if 'cel_2' in col.lower()]
        
        if clean_cell_columns and dirty_cell_columns:
            clean_irradiance = data[clean_cell_columns].mean(axis=1)
            dirty_irradiance = data[dirty_cell_columns].mean(axis=1)
            
            # Soiling ratio
            soiling_ratio = dirty_irradiance / (clean_irradiance + 1e-6)
            soiling_losses = theoretical_gen * (1 - soiling_ratio)
            return np.maximum(soiling_losses, 0)
        
        # Method 2: Seasonal soiling model
        if 'day_of_year' in data.columns:
            # Soiling typically accumulates over time and is cleaned periodically
            # Assume cleaning every 30 days
            days_since_cleaning = data['day_of_year'] % 30
            soiling_factor = days_since_cleaning / 30 * 0.05  # 5% max soiling loss
            soiling_losses = theoretical_gen * soiling_factor
            return np.maximum(soiling_losses, 0)
        
        # Method 3: Humidity and wind based model
        if 'avg_humidity' in data.columns and 'avg_wind_speed' in data.columns:
            # Low wind and high humidity increase soiling
            wind_factor = np.exp(-data['avg_wind_speed'] / 5)  # Higher wind reduces soiling
            humidity_factor = data['avg_humidity'] / 100  # Higher humidity increases soiling
            soiling_factor = wind_factor * humidity_factor * 0.03
            soiling_losses = theoretical_gen * soiling_factor
            return np.maximum(soiling_losses, 0)
        
        # Method 4: Time-based accumulation model
        if 'datetime' in data.columns:
            # Simple accumulation model
            soiling_factor = 0.02  # 2% base soiling loss
            soiling_losses = theoretical_gen * soiling_factor
            return np.maximum(soiling_losses, 0)
        
        return pd.Series([0.0] * len(data))
    
    def _aggregate_results(self, data, loss_attribution, time_granularity):
        """
        Aggregate and format results
        
        Args:
            data (pd.DataFrame): Analysis data
            loss_attribution (dict): Loss attribution results
            time_granularity (str): Time granularity
            
        Returns:
            dict: Formatted results
        """
        results = {}
        
        # Create loss breakdown DataFrame
        loss_breakdown = pd.DataFrame({
            'Cloud Cover': loss_attribution['cloud_losses'],
            'Shading': loss_attribution['shading_losses'],
            'Temperature': loss_attribution['temperature_losses'],
            'Soiling': loss_attribution['soiling_losses'],
            'Other Losses': loss_attribution['other_losses']
        })
        
        if 'datetime' in data.columns:
            loss_breakdown.index = data['datetime']
        
        results['loss_breakdown'] = loss_breakdown
        
        # Calculate summary statistics
        loss_summary = pd.DataFrame({
            'Loss Type': ['Cloud Cover', 'Shading', 'Temperature', 'Soiling', 'Other Losses'],
            'Total Loss (kWh)': [
                loss_attribution['cloud_losses'].sum(),
                loss_attribution['shading_losses'].sum(),
                loss_attribution['temperature_losses'].sum(),
                loss_attribution['soiling_losses'].sum(),
                loss_attribution['other_losses'].sum()
            ],
            'Average Loss (kWh)': [
                loss_attribution['cloud_losses'].mean(),
                loss_attribution['shading_losses'].mean(),
                loss_attribution['temperature_losses'].mean(),
                loss_attribution['soiling_losses'].mean(),
                loss_attribution['other_losses'].mean()
            ],
            'Percentage of Total': [
                loss_attribution['cloud_losses'].sum() / (loss_attribution['cloud_losses'].sum() + 
                                                         loss_attribution['shading_losses'].sum() + 
                                                         loss_attribution['temperature_losses'].sum() + 
                                                         loss_attribution['soiling_losses'].sum() + 
                                                         loss_attribution['other_losses'].sum()) * 100
                for _ in range(5)
            ]
        })
        
        # Calculate actual percentages
        total_losses = sum([loss_attribution[key].sum() for key in loss_attribution.keys()])
        if total_losses > 0:
            loss_summary['Percentage of Total'] = [
                loss_attribution['cloud_losses'].sum() / total_losses * 100,
                loss_attribution['shading_losses'].sum() / total_losses * 100,
                loss_attribution['temperature_losses'].sum() / total_losses * 100,
                loss_attribution['soiling_losses'].sum() / total_losses * 100,
                loss_attribution['other_losses'].sum() / total_losses * 100
            ]
        
        results['loss_summary'] = loss_summary
        
        # Create trend analysis
        if 'datetime' in data.columns and len(data) > 1:
            trends = loss_breakdown.copy()
            trends['datetime'] = data['datetime']
            results['trends'] = trends
        
        return results
    
    def generate_loss_report(self, results, output_path=None):
        """
        Generate comprehensive loss attribution report
        
        Args:
            results (dict): Loss attribution results
            output_path (str): Path to save report
            
        Returns:
            str: Report content
        """
        report_content = []
        report_content.append("# Solar Energy Loss Attribution Report\n")
        report_content.append(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        # Summary statistics
        if 'loss_summary' in results:
            report_content.append("## Loss Summary\n")
            report_content.append(results['loss_summary'].to_string(index=False))
            report_content.append("\n\n")
        
        # Detailed breakdown
        if 'loss_breakdown' in results:
            report_content.append("## Detailed Loss Breakdown\n")
            report_content.append(results['loss_breakdown'].describe().to_string())
            report_content.append("\n\n")
        
        # Trend analysis
        if 'trends' in results:
            report_content.append("## Trend Analysis\n")
            report_content.append("Time series analysis of loss patterns over the analysis period.\n")
            report_content.append("\n")
        
        report_text = "".join(report_content)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text
