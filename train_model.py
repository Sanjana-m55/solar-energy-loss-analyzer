import os
# Set environment variables before importing ML libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from utils import setup_logging

class AdvancedMLPipeline:
    """Advanced ML pipeline for theoretical generation modeling"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def train_models(self, X, y):
        """Train multiple advanced ML models"""
        print("Training advanced ML models...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define advanced models with optimized parameters
        models = {
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=80,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.05,
                min_child_weight=1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=1
            )
        }
        
        # Train individual models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=1)
                
                # Train on full dataset
                model.fit(X, y)
                
                # Predictions
                y_pred = model.predict(X)
                
                # Metrics
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                
                self.models[name] = model
                self.model_performance[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'model': model
                }
                
                print(f"   CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"   Full Data R² Score: {r2:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                
            except Exception as e:
                print(f"   Error training {name}: {str(e)}")
                continue
        
        # Train ensemble models
        self._train_ensemble_models(X, y, tscv)
        
        # Select best model
        self._select_best_model()
        
        return self.model_performance
    
    def _train_ensemble_models(self, X, y, tscv):
        """Train advanced ensemble models"""
        print("\nTraining ensemble models...")
        
        if len(self.models) < 2:
            print("   Not enough base models for ensemble")
            return
        
        try:
            # Voting Regressor
            base_models = [(name, model) for name, model in self.models.items()]
            voting_model = VotingRegressor(estimators=base_models)
            
            # Cross-validation for ensemble
            cv_scores = cross_val_score(voting_model, X, y, cv=tscv, scoring='r2', n_jobs=1)
            
            # Train ensemble
            voting_model.fit(X, y)
            y_pred_ensemble = voting_model.predict(X)
            
            # Ensemble metrics
            r2_ensemble = r2_score(y, y_pred_ensemble)
            rmse_ensemble = np.sqrt(mean_squared_error(y, y_pred_ensemble))
            mae_ensemble = mean_absolute_error(y, y_pred_ensemble)
            
            self.models['Ensemble_Voting'] = voting_model
            self.model_performance['Ensemble_Voting'] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'r2_score': r2_ensemble,
                'rmse': rmse_ensemble,
                'mae': mae_ensemble,
                'model': voting_model
            }
            
            print(f"   Voting Ensemble CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"   Voting Ensemble Full Data R² Score: {r2_ensemble:.4f}")
            
        except Exception as e:
            print(f"   Error training ensemble: {str(e)}")
    
    def _select_best_model(self):
        """Select the best performing model"""
        if not self.model_performance:
            return
        
        # Select based on cross-validation performance
        best_name = max(self.model_performance.keys(), 
                       key=lambda x: self.model_performance[x]['cv_mean'])
        
        self.best_model = self.model_performance[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\nBest Model Selected: {best_name}")
        print(f"CV R² Score: {self.model_performance[best_name]['cv_mean']:.4f}")
    
    def get_feature_importance(self):
        """Get feature importance from best model"""
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        return self.best_model.predict(X)

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("ADVANCED SOLAR ENERGY LOSS ANALYSIS - ML MODEL TRAINING")
    print("=" * 70)
    
    # Initialize components
    processor = DataProcessor()
    pipeline = AdvancedMLPipeline()
    logger = setup_logging()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and process data
    print("\n1. Loading and processing solar PV data...")
    try:
        raw_data = processor.load_data('data/data.csv')
        print(f"   Raw data loaded: {raw_data.shape}")
        
        processed_data = processor.preprocess_data(raw_data)
        print(f"   Data processed: {processed_data.shape}")
        print(f"   Time range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}")
        
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        return False
    
    # Prepare ML features
    print("\n2. Engineering features for advanced ML...")
    try:
        # Identify target variable
        energy_cols = [col for col in processed_data.columns if 'energy' in col.lower()]
        if not energy_cols:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            target_candidates = [col for col in numeric_cols if any(kw in col.lower() 
                               for kw in ['power', 'generation', 'output', 'kwh', 'mwh'])]
            target_column = target_candidates[0] if target_candidates else numeric_cols[0]
        else:
            target_column = energy_cols[0]
        
        print(f"   Target variable: {target_column}")
        
        # Prepare features
        X, y = processor.prepare_ml_features(processed_data, target_column)
        pipeline.feature_names = processor.get_feature_names()
        
        print(f"   Features engineered: {X.shape}")
        print(f"   Target samples: {y.shape}")
        print(f"   Feature count: {len(pipeline.feature_names)}")
        
        # Data quality check
        print(f"   Missing values in features: {X.isnull().sum().sum()}")
        print(f"   Missing values in target: {y.isnull().sum()}")
        
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        return False
    
    # Train advanced ML models
    print("\n3. Training advanced ML pipeline...")
    try:
        results = pipeline.train_models(X, y)
        
        if not results:
            print("   ERROR: No models trained successfully")
            return False
        
        print(f"   Successfully trained {len(results)} models")
        
    except Exception as e:
        print(f"   ERROR: {str(e)}")
        return False
    
    # Model evaluation and comparison
    print("\n4. Model evaluation and comparison...")
    
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'CV_R2_Mean': metrics['cv_mean'],
            'CV_R2_Std': metrics['cv_std'],
            'Full_Data_R2': metrics['r2_score'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('CV_R2_Mean', ascending=False)
    
    print("   Model Performance Comparison:")
    print(comparison_df.round(4).to_string(index=False))
    
    # Feature importance analysis
    print("\n5. Feature importance analysis...")
    feature_importance = pipeline.get_feature_importance()
    
    if feature_importance is not None:
        print("   Top 10 Most Important Features:")
        print(feature_importance.head(10).round(4).to_string(index=False))
    
    # Save model artifacts
    print("\n6. Saving model artifacts...")
    try:
        # Save best model
        model_path = 'models/best_theoretical_model.pkl'
        joblib.dump(pipeline.best_model, model_path)
        print(f"   Best model saved: {model_path}")
        
        # Save data processor
        processor_path = 'models/data_processor.pkl'
        joblib.dump(processor, processor_path)
        print(f"   Data processor saved: {processor_path}")
        
        # Save feature importance
        if feature_importance is not None:
            feature_importance.to_csv('models/feature_importance.csv', index=False)
            print(f"   Feature importance saved: models/feature_importance.csv")
        
        # Save comprehensive metadata
        metadata = {
            'model_name': pipeline.best_model_name,
            'model_type': str(type(pipeline.best_model)),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_column': target_column,
            'feature_names': pipeline.feature_names,
            'n_features': len(pipeline.feature_names),
            'n_training_samples': len(X),
            'data_shape': {
                'raw': list(raw_data.shape),
                'processed': list(processed_data.shape),
                'features': list(X.shape),
                'target': list(y.shape)
            },
            'performance_metrics': results[pipeline.best_model_name],
            'all_model_results': {name: {
                'cv_mean': res['cv_mean'],
                'cv_std': res['cv_std'],
                'r2_score': res['r2_score'],
                'rmse': res['rmse'],
                'mae': res['mae']
            } for name, res in results.items()},
            'training_config': {
                'cross_validation_folds': 5,
                'ensemble_methods': 'Voting Regressor',
                'feature_engineering': 'Advanced solar-specific features',
                'model_selection': 'Cross-validation based'
            }
        }
        
        metadata_path = 'models/model_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        print(f"   Model metadata saved: {metadata_path}")
        
        # Save comparison results
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        print(f"   Model comparison saved: models/model_comparison.csv")
        
    except Exception as e:
        print(f"   ERROR saving artifacts: {str(e)}")
        return False
    
    # Success summary
    print("\n" + "=" * 70)
    print("ADVANCED ML TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Best Model: {pipeline.best_model_name}")
    print(f"Cross-Validation R² Score: {results[pipeline.best_model_name]['cv_mean']:.4f}")
    print(f"Full Dataset R² Score: {results[pipeline.best_model_name]['r2_score']:.4f}")
    print(f"RMSE: {results[pipeline.best_model_name]['rmse']:.4f}")
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModel artifacts saved to 'models/' directory:")
    print("  • best_theoretical_model.pkl (trained model)")
    print("  • data_processor.pkl (preprocessing pipeline)")
    print("  • model_metadata.pkl (comprehensive metadata)")
    print("  • model_comparison.csv (performance comparison)")
    print("  • feature_importance.csv (feature rankings)")
    print("\nReady for deployment in Streamlit application!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTraining failed. Check error messages above.")
        exit(1)
    else:
        print("\nNext step: Start Streamlit app with trained model")
        print("Command: streamlit run main_app.py --server.port 5000")