import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """
    Advanced ML Pipeline for theoretical generation modeling
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ML pipeline
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_params = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def prepare_features(self, data):
        """
        Prepare features and target for ML modeling
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            tuple: (X, y) features and target
        """
        # Identify potential target columns (actual generation)
        target_candidates = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['eact', 'p_tot', 'export', 'power']):
                if data[col].dtype in ['int64', 'float64']:
                    target_candidates.append(col)
        
        # Use the first suitable target or create a synthetic one
        if target_candidates:
            target_column = target_candidates[0]
            y = data[target_column].copy()
        else:
            # Create synthetic target based on available power data
            power_columns = [col for col in data.columns if 'p' in col.lower() and data[col].dtype in ['int64', 'float64']]
            if power_columns:
                y = data[power_columns].sum(axis=1)
            else:
                # Use a combination of irradiance and other factors as proxy
                irradiance_cols = [col for col in data.columns if 'gii' in col.lower() or 'ghi' in col.lower()]
                if irradiance_cols:
                    y = data[irradiance_cols].mean(axis=1) * 0.001  # Scale to reasonable power values
                else:
                    raise ValueError("No suitable target variable found")
        
        # Prepare features
        exclude_columns = ['datetime', 'zone', 'inverter', 'string', 'season'] + target_candidates
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        feature_columns = [col for col in feature_columns if data[col].dtype in ['int64', 'float64']]
        
        X = data[feature_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        return X, y
    
    def train_model(self, X, y, use_ensemble=True, optimize_hyperparams=True):
        """
        Train the theoretical generation model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            use_ensemble (bool): Whether to use ensemble methods
            optimize_hyperparams (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training results and performance metrics
        """
        print("Starting model training...")
        
        # Split data for time series
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Individual model training
        self._train_baseline_models(X, y, tscv)
        
        # Ensemble model training
        if use_ensemble:
            self._train_ensemble_models(X, y, tscv)
        
        # Select best model
        self._select_best_model(X, y)
        
        # Generate predictions for evaluation
        y_pred = self.best_model.predict(X)
        
        # Calculate performance metrics
        results = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'model_name': self.best_model.__class__.__name__,
            'model_comparison': self._get_model_comparison(),
            'feature_importance': self._get_feature_importance(X.columns),
            'predictions': y_pred,
            'actual': y
        }
        
        print(f"Best model: {results['model_name']}")
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        
        return results
    
    def _train_baseline_models(self, X, y, tscv):
        """
        Train baseline models without hyperparameter optimization
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            tscv: Time series cross-validator
        """
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            random_state=self.random_state,
            verbose=-1
        )
        self.models['LightGBM'] = lgb_model
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            random_state=self.random_state,
            verbosity=0
        )
        self.models['XGBoost'] = xgb_model
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state
        )
        self.models['RandomForest'] = rf_model
        
        # CatBoost (if available)
        if cb is not None:
            cat_model = cb.CatBoostRegressor(
                iterations=100,
                random_state=self.random_state,
                verbose=False
            )
            self.models['CatBoost'] = cat_model
        
        # Train and evaluate each model
        for name, model in self.models.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                
                # Fit model
                model.fit(X, y)
                
                # Store performance
                self.model_performance[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }
                
                print(f"{name} - CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
    
    def _train_with_optuna(self, X, y, tscv):
        """
        Placeholder for Optuna optimization - using baseline models instead
        """
        print("Optuna not available - using baseline models...")
        self._train_baseline_models(X, y, tscv)
    
    def _optimize_lightgbm(self, X, y, tscv):
        """Optimize LightGBM hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # Train best model
        best_model = lgb.LGBMRegressor(**study.best_params)
        best_model.fit(X, y)
        
        self.models['LightGBM_Optimized'] = best_model
        self.best_params['LightGBM'] = study.best_params
        self.model_performance['LightGBM_Optimized'] = {
            'cv_mean': study.best_value,
            'cv_std': 0,  # Optuna doesn't provide std
            'model': best_model
        }
        
        print(f"LightGBM Optimized - Best CV R² Score: {study.best_value:.4f}")
    
    def _optimize_xgboost(self, X, y, tscv):
        """Optimize XGBoost hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # Train best model
        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(X, y)
        
        self.models['XGBoost_Optimized'] = best_model
        self.best_params['XGBoost'] = study.best_params
        self.model_performance['XGBoost_Optimized'] = {
            'cv_mean': study.best_value,
            'cv_std': 0,
            'model': best_model
        }
        
        print(f"XGBoost Optimized - Best CV R² Score: {study.best_value:.4f}")
    
    def _optimize_random_forest(self, X, y, tscv):
        """Optimize Random Forest hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'random_state': self.random_state
            }
            
            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        # Train best model
        best_model = RandomForestRegressor(**study.best_params)
        best_model.fit(X, y)
        
        self.models['RandomForest_Optimized'] = best_model
        self.best_params['RandomForest'] = study.best_params
        self.model_performance['RandomForest_Optimized'] = {
            'cv_mean': study.best_value,
            'cv_std': 0,
            'model': best_model
        }
        
        print(f"Random Forest Optimized - Best CV R² Score: {study.best_value:.4f}")
    
    def _optimize_catboost(self, X, y, tscv):
        """Optimize CatBoost hyperparameters"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = cb.CatBoostRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        # Train best model
        best_model = cb.CatBoostRegressor(**study.best_params)
        best_model.fit(X, y)
        
        self.models['CatBoost_Optimized'] = best_model
        self.best_params['CatBoost'] = study.best_params
        self.model_performance['CatBoost_Optimized'] = {
            'cv_mean': study.best_value,
            'cv_std': 0,
            'model': best_model
        }
        
        print(f"CatBoost Optimized - Best CV R² Score: {study.best_value:.4f}")
    
    def _train_ensemble_models(self, X, y, tscv):
        """
        Train ensemble models using voting and stacking
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            tscv: Time series cross-validator
        """
        print("Training ensemble models...")
        
        # Get top 3 models
        top_models = sorted(self.model_performance.items(), 
                          key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        
        # Voting Ensemble
        voting_estimators = [(name, perf['model']) for name, perf in top_models]
        voting_regressor = VotingRegressor(voting_estimators)
        
        # Cross-validation for voting ensemble
        cv_scores = cross_val_score(voting_regressor, X, y, cv=tscv, scoring='r2')
        voting_regressor.fit(X, y)
        
        self.models['VotingEnsemble'] = voting_regressor
        self.model_performance['VotingEnsemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': voting_regressor
        }
        
        print(f"Voting Ensemble - CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Stacking Ensemble
        self._train_stacking_ensemble(X, y, tscv, top_models)
    
    def _train_stacking_ensemble(self, X, y, tscv, base_models):
        """
        Train stacking ensemble with meta-learner
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            tscv: Time series cross-validator
            base_models (list): List of base models
        """
        try:
            # Generate meta-features using cross-validation
            meta_features = np.zeros((len(X), len(base_models)))
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                for i, (name, perf) in enumerate(base_models):
                    # Clone and train model
                    model = perf['model'].__class__(**self.best_params.get(name.split('_')[0], {}))
                    model.fit(X_train, y_train)
                    
                    # Generate predictions for validation set
                    pred = model.predict(X_val)
                    meta_features[val_idx, i] = pred
            
            # Train meta-learner
            meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
            cv_scores = cross_val_score(meta_learner, meta_features, y, cv=tscv, scoring='r2')
            meta_learner.fit(meta_features, y)
            
            # Create stacking ensemble
            class StackingEnsemble:
                def __init__(self, base_models, meta_learner):
                    self.base_models = base_models
                    self.meta_learner = meta_learner
                
                def predict(self, X):
                    # Generate predictions from base models
                    base_predictions = np.zeros((len(X), len(self.base_models)))
                    for i, (name, perf) in enumerate(self.base_models):
                        base_predictions[:, i] = perf['model'].predict(X)
                    
                    # Meta-learner prediction
                    return self.meta_learner.predict(base_predictions)
            
            stacking_ensemble = StackingEnsemble(base_models, meta_learner)
            
            self.models['StackingEnsemble'] = stacking_ensemble
            self.model_performance['StackingEnsemble'] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': stacking_ensemble
            }
            
            print(f"Stacking Ensemble - CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
        except Exception as e:
            print(f"Error training stacking ensemble: {str(e)}")
    
    def _select_best_model(self, X, y):
        """
        Select the best performing model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
        """
        best_score = -np.inf
        best_name = None
        
        for name, perf in self.model_performance.items():
            if perf['cv_mean'] > best_score:
                best_score = perf['cv_mean']
                best_name = name
        
        self.best_model = self.model_performance[best_name]['model']
        print(f"Selected best model: {best_name} (CV R² Score: {best_score:.4f})")
    
    def _get_model_comparison(self):
        """
        Get model comparison DataFrame
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        comparison_data = []
        for name, perf in self.model_performance.items():
            comparison_data.append({
                'Model': name,
                'CV_R2_Mean': perf['cv_mean'],
                'CV_R2_Std': perf['cv_std']
            })
        
        return pd.DataFrame(comparison_data).sort_values('CV_R2_Mean', ascending=False)
    
    def _get_feature_importance(self, feature_names):
        """
        Get feature importance from the best model
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance
        """
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                return importance_df
            else:
                return pd.DataFrame({'Feature': feature_names, 'Importance': [0] * len(feature_names)})
        except:
            return pd.DataFrame({'Feature': feature_names, 'Importance': [0] * len(feature_names)})
    
    def predict(self, X):
        """
        Make predictions using the best model
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict(X)
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
