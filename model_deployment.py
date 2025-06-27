import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelDeployment:
    """
    Model deployment class for loading and using pre-trained models
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the model deployment
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.model = None
        self.processor = None
        self.metadata = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_trained_model(self):
        """
        Load the pre-trained model and components
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, 'best_theoretical_model.pkl')
            processor_path = os.path.join(self.models_dir, 'data_processor.pkl')
            metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
            
            # Check if files exist
            if not all(os.path.exists(path) for path in [model_path, processor_path, metadata_path]):
                missing_files = [path for path in [model_path, processor_path, metadata_path] 
                               if not os.path.exists(path)]
                print(f"Missing model files: {missing_files}")
                return False
            
            # Load components
            self.model = joblib.load(model_path)
            self.processor = joblib.load(processor_path)
            self.metadata = joblib.load(metadata_path)
            
            # Extract feature names
            self.feature_names = self.metadata['feature_names']
            
            self.is_loaded = True
            print(f"Successfully loaded {self.metadata['model_name']} model")
            print(f"Training date: {self.metadata['training_date']}")
            print(f"Model performance - R² Score: {self.metadata['performance_metrics']['full_data_r2']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def check_model_availability(self):
        """
        Check if pre-trained model is available
        
        Returns:
            dict: Status information about model availability
        """
        model_files = {
            'best_theoretical_model.pkl': 'Trained ML model',
            'data_processor.pkl': 'Data preprocessing pipeline',
            'model_metadata.pkl': 'Model information and metrics'
        }
        
        status = {
            'available': True,
            'missing_files': [],
            'file_status': {}
        }
        
        for filename, description in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            exists = os.path.exists(filepath)
            status['file_status'][filename] = {
                'exists': exists,
                'description': description,
                'path': filepath
            }
            
            if not exists:
                status['missing_files'].append(filename)
                status['available'] = False
        
        return status
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information and performance metrics
        """
        if not self.is_loaded:
            return {"error": "Model not loaded. Please train the model first."}
        
        return {
            "model_name": self.metadata['model_name'],
            "model_type": self.metadata['model_type'],
            "training_date": self.metadata['training_date'],
            "target_column": self.metadata['target_column'],
            "n_features": self.metadata['n_features'],
            "n_training_samples": self.metadata['n_training_samples'],
            "performance_metrics": self.metadata['performance_metrics'],
            "feature_names": self.feature_names[:10]  # Show first 10 features
        }
    
    def get_model_comparison(self):
        """
        Get comparison of all trained models
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.is_loaded:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, results in self.metadata['all_model_results'].items():
            comparison_data.append({
                'Model': model_name,
                'CV_R2_Mean': results['cv_mean'],
                'CV_R2_Std': results['cv_std'],
                'Full_Data_R2': results['r2_score'],
                'RMSE': results['rmse'],
                'MAE': results['mae']
            })
        
        return pd.DataFrame(comparison_data).sort_values('CV_R2_Mean', ascending=False)
    
    def get_feature_importance(self):
        """
        Get feature importance if available
        
        Returns:
            pd.DataFrame: Feature importance data
        """
        importance_path = os.path.join(self.models_dir, 'feature_importance.csv')
        if os.path.exists(importance_path):
            return pd.read_csv(importance_path)
        else:
            return pd.DataFrame()
    
    def predict_theoretical_generation(self, data):
        """
        Predict theoretical generation using the pre-trained model
        
        Args:
            data (pd.DataFrame): Processed solar data
            
        Returns:
            np.array: Theoretical generation predictions
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Please train the model first.")
        
        try:
            # Prepare features using the loaded processor
            X, _ = self.processor.prepare_ml_features(data, self.metadata['target_column'])
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def process_data(self, raw_data):
        """
        Process raw data using the saved processor
        
        Args:
            raw_data (pd.DataFrame): Raw solar data
            
        Returns:
            pd.DataFrame: Processed data
        """
        if not self.is_loaded:
            raise ValueError("Processor not loaded. Please train the model first.")
        
        return self.processor.preprocess_data(raw_data)
    
    def load_data(self, file_path):
        """
        Load data using the saved processor
        
        Args:
            file_path (str): Path to data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.is_loaded:
            raise ValueError("Processor not loaded. Please train the model first.")
        
        return self.processor.load_data(file_path)
    
    def create_training_instructions(self):
        """
        Create instructions for training the model
        
        Returns:
            str: Training instructions
        """
        return """
        # Model Training Instructions
        
        To train the theoretical generation model:
        
        1. Open and run the Jupyter notebook: `model_training.ipynb`
        2. The notebook will:
           - Load and preprocess the solar data
           - Train multiple ML models (LightGBM, XGBoost, Random Forest)
           - Select the best performing model
           - Save the trained model as .pkl files
        
        3. Required files will be saved in 'models/' directory:
           - best_theoretical_model.pkl (trained model)
           - data_processor.pkl (preprocessing pipeline)
           - model_metadata.pkl (model information)
           - model_comparison.csv (performance comparison)
           - feature_importance.csv (feature rankings)
        
        4. After training, refresh the Streamlit app to load the new model
        
        The training process typically takes 5-15 minutes depending on data size.
        """

# Global instance for the Streamlit app
model_deployment = ModelDeployment()