"""
Advanced PJM Day-Ahead Price Prediction with 2-Stage Spike Detection
=====================================================================
Implements:
- Two-stage spike prediction (binary classification then magnitude)
- Separate peak/off-peak spike thresholds
- Quantile Regression Averaging (QRA)
- Conformal prediction for confidence intervals
- Dynamic ensemble selection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, Dict, List, Optional
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# Default threads
DEFAULT_NJOBS = max(1, os.cpu_count() or 1)
IS_DARWIN = (platform.system() == 'Darwin')

# =============================================================================
# TWO-STAGE SPIKE DETECTOR
# =============================================================================

class TwoStageSpikePrediction:
    """
    Two-stage spike prediction:
    Stage 1: Binary classification (spike yes/no)
    Stage 2: Magnitude prediction for predicted spikes
    """
    
    def __init__(self, 
                 peak_threshold: float = 175,
                 offpeak_threshold: float = 100,
                 n_jobs: int = DEFAULT_NJOBS):
        """
        Parameters:
        -----------
        peak_threshold : float
            Spike threshold for peak hours (7am-11pm weekdays)
        offpeak_threshold : float
            Spike threshold for off-peak hours
        """
        self.peak_threshold = peak_threshold
        self.offpeak_threshold = offpeak_threshold
        self.n_jobs = (n_jobs if (isinstance(n_jobs, int) and n_jobs > 0) else DEFAULT_NJOBS)
        
        # Stage 1: Binary classifiers
        self.spike_classifier = None
        
        # Stage 2: Magnitude predictors (separate for spike/non-spike)
        self.spike_magnitude_model = None
        self.normal_price_model = None
        
        # Feature importance
        self.spike_features = None
        self.magnitude_features = None
        
    def _get_dynamic_threshold(self, X: pd.DataFrame) -> np.ndarray:
        """Get threshold based on peak/off-peak hours"""
        thresholds = np.ones(len(X)) * self.offpeak_threshold
        
        # Identify peak hours
        if 'hour' in X.columns and 'is_weekend' in X.columns:
            is_peak = ((X['hour'] >= 7) & (X['hour'] < 23) & (X['is_weekend'] == 0))
            thresholds[is_peak] = self.peak_threshold
        elif 'is_peak' in X.columns:
            thresholds[X['is_peak'] == 1] = self.peak_threshold
            
        return thresholds
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        """
        Fit two-stage model
        """
        print("Training 2-Stage Spike Detector...")
        
        # Get dynamic thresholds
        thresholds = self._get_dynamic_threshold(X)
        
        # Create spike labels
        y_spike = (y.values > thresholds).astype(int)
        
        # ========== STAGE 1: Binary Classification ==========
        print("  Stage 1: Training spike classifier...")
        
        # Use XGBoost for binary classification with class weights
        spike_ratio = y_spike.mean()
        scale_pos_weight = (1 - spike_ratio) / max(spike_ratio, 0.001)
        
        self.spike_classifier = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            scale_pos_weight=scale_pos_weight,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=2,
            min_child_weight=3,
            n_jobs=self.n_jobs,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Fit with sample weights if provided
        if sample_weight is not None:
            self.spike_classifier.fit(X, y_spike, sample_weight=sample_weight)
        else:
            self.spike_classifier.fit(X, y_spike)
        
        # Store feature importance
        self.spike_features = pd.DataFrame({
            'feature': X.columns,
            'importance': self.spike_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"    Spike rate: {spike_ratio:.1%}")
        print(f"    Top spike features: {self.spike_features.head(5)['feature'].tolist()}")
        
        # ========== STAGE 2: Magnitude Prediction ==========
        print("  Stage 2: Training magnitude models...")
        
        # Separate spike and normal observations
        spike_mask = y_spike == 1
        normal_mask = ~spike_mask
        
        # Train spike magnitude model (for when we predict a spike)
        if spike_mask.sum() > 10:
            print(f"    Training spike magnitude model on {spike_mask.sum()} samples...")
            
            self.spike_magnitude_model = lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=31,
                min_child_weight=5,
                reg_alpha=0.2,
                reg_lambda=1,
                n_jobs=self.n_jobs,
                random_state=42,
                verbosity=-1
            ,
                device_type='cpu'
            )
            
            # Use weights to emphasize extreme spikes
            spike_weights = None
            if sample_weight is not None:
                spike_weights = sample_weight[spike_mask]
                
            # Add extra weight for extreme spikes
            spike_prices = y[spike_mask]
            spike_90th = spike_prices.quantile(0.9)
            if spike_weights is None:
                spike_weights = np.ones(spike_mask.sum())
            spike_weights = spike_weights * (1 + (spike_prices > spike_90th).astype(float))
            
            self.spike_magnitude_model.fit(
                X[spike_mask], 
                y[spike_mask],
                sample_weight=spike_weights
            )
        
        # Train normal price model (for when we predict no spike)
        print(f"    Training normal price model on {normal_mask.sum()} samples...")
        
        self.normal_price_model = lgb.LGBMRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=31,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1,
            n_jobs=self.n_jobs,
            random_state=42,
            verbosity=-1
        ,
                device_type='cpu'
            )
        
        normal_weights = sample_weight[normal_mask] if sample_weight is not None else None
        self.normal_price_model.fit(
            X[normal_mask], 
            y[normal_mask],
            sample_weight=normal_weights
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict prices using two-stage approach
        
        Returns:
        --------
        prices : np.ndarray
            Predicted prices
        spike_probs : np.ndarray
            Spike probabilities
        """
        
        # Stage 1: Predict spike probability
        spike_probs = self.spike_classifier.predict_proba(X)[:, 1]
        spike_binary = (spike_probs > 0.5).astype(int)
        
        # Stage 2: Predict magnitude based on spike prediction
        prices = np.zeros(len(X))
        
        # For predicted spikes, use spike magnitude model
        spike_mask = spike_binary == 1
        if spike_mask.sum() > 0 and self.spike_magnitude_model is not None:
            prices[spike_mask] = self.spike_magnitude_model.predict(X[spike_mask])
        
        # For predicted normal, use normal price model
        normal_mask = ~spike_mask
        if normal_mask.sum() > 0:
            prices[normal_mask] = self.normal_price_model.predict(X[normal_mask])
        
        # Apply dynamic thresholds as minimum for spikes
        thresholds = self._get_dynamic_threshold(X)
        prices[spike_mask] = np.maximum(prices[spike_mask], thresholds[spike_mask])
        
        return prices, spike_probs


# =============================================================================
# QUANTILE REGRESSION AVERAGING (QRA)
# =============================================================================

class QuantileRegressionAveraging:
    """
    Implements Quantile Regression Averaging for probabilistic forecasting
    """
    
    def __init__(self, 
                 quantiles: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                 n_jobs: int = -1):
        """
        Parameters:
        -----------
        quantiles : list
            Quantiles to predict
        """
        self.quantiles = quantiles
        self.n_jobs = n_jobs
        self.models = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit quantile regression models
        """
        print(f"Training QRA with {len(self.quantiles)} quantiles...")
        
        for q in self.quantiles:
            print(f"  Training quantile {q:.2f}...")
            
            # Use LightGBM with quantile loss
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=31,
                min_child_weight=10,
                n_jobs=self.n_jobs,
                random_state=42,
                verbosity=-1
            ,
                device_type='cpu'
            )
            
            model.fit(X, y)
            self.models[q] = model
            
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions
        
        Returns:
        --------
        Dictionary with quantile values as keys and predictions as values
        """
        predictions = {}
        
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
            
        return predictions
    
    def predict_intervals(self, X: pd.DataFrame, coverage: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict intervals with specified coverage
        
        Returns:
        --------
        median : np.ndarray
            Median predictions (50th percentile)
        lower : np.ndarray
            Lower bound
        upper : np.ndarray
            Upper bound
        """
        
        # Calculate quantiles for interval
        lower_q = (1 - coverage) / 2
        upper_q = 1 - lower_q
        
        # Get predictions
        preds = self.predict(X)
        
        # Find closest quantiles
        lower_quantile = min(self.quantiles, key=lambda x: abs(x - lower_q))
        upper_quantile = min(self.quantiles, key=lambda x: abs(x - upper_q))
        
        return preds[0.5], preds[lower_quantile], preds[upper_quantile]


# =============================================================================
# CONFORMAL PREDICTION
# =============================================================================

class ConformalPredictor:
    """
    Implements conformal prediction for valid confidence intervals
    """
    
    def __init__(self, base_model, coverage: float = 0.9):
        """
        Parameters:
        -----------
        base_model : estimator
            Base prediction model
        coverage : float
            Desired coverage level (e.g., 0.9 for 90% intervals)
        """
        self.base_model = base_model
        self.coverage = coverage
        self.calibration_scores = None
        self.quantile = None
        
    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Calibrate conformal predictor on calibration set
        """
        print(f"Calibrating conformal predictor for {self.coverage:.0%} coverage...")
        
        # Get predictions on calibration set
        predictions = self.base_model.predict(X_cal)
        
        # Calculate nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal.values - predictions)
        
        # Calculate quantile for desired coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * self.coverage) / n
        q_level = min(q_level, 1.0)
        
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        print(f"  Calibration quantile: {self.quantile:.2f}")
        print(f"  Mean calibration error: {self.calibration_scores.mean():.2f}")
        
        return self
    
    def predict_interval(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals
        
        Returns:
        --------
        predictions : np.ndarray
            Point predictions
        lower : np.ndarray
            Lower confidence bound
        upper : np.ndarray
            Upper confidence bound
        """
        
        if self.quantile is None:
            raise ValueError("Model must be calibrated before prediction")
        
        # Get point predictions
        predictions = self.base_model.predict(X)
        
        # Create intervals using calibrated quantile
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        # Ensure non-negative prices
        lower = np.maximum(lower, 0)
        
        return predictions, lower, upper
    
    def evaluate_coverage(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate actual coverage on test set
        """
        predictions, lower, upper = self.predict_interval(X_test)
        
        # Check coverage
        covered = (y_test.values >= lower) & (y_test.values <= upper)
        actual_coverage = covered.mean()
        
        # Calculate interval width
        interval_width = upper - lower
        
        return {
            'actual_coverage': actual_coverage,
            'desired_coverage': self.coverage,
            'coverage_gap': actual_coverage - self.coverage,
            'mean_interval_width': interval_width.mean(),
            'median_interval_width': np.median(interval_width)
        }


# =============================================================================
# DYNAMIC ENSEMBLE WITH THOMPSON SAMPLING
# =============================================================================

class ThompsonSamplingEnsemble:
    """
    Dynamic ensemble selection using Thompson Sampling
    """
    
    def __init__(self, models: Dict, window_size: int = 100):
        """
        Parameters:
        -----------
        models : dict
            Dictionary of model_name: model pairs
        window_size : int
            Window for tracking recent performance
        """
        self.models = models
        self.window_size = window_size
        
        # Track performance with Beta distribution parameters
        self.successes = {name: 1 for name in models.keys()}
        self.failures = {name: 1 for name in models.keys()}
        
        # Recent predictions buffer
        self.recent_predictions = []
        self.recent_actuals = []
        
    def select_model(self) -> str:
        """
        Select model using Thompson Sampling
        """
        # Sample from Beta distribution for each model
        samples = {}
        for name in self.models.keys():
            samples[name] = np.random.beta(
                self.successes[name],
                self.failures[name]
            )
        
        # Select model with highest sample
        return max(samples, key=samples.get)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Thompson Sampling selection
        """
        predictions = np.zeros(len(X))
        selected_models = []
        
        for i in range(len(X)):
            # Select model for this prediction
            model_name = self.select_model()
            selected_models.append(model_name)
            
            # Make prediction
            X_i = X.iloc[[i]]
            if hasattr(self.models[model_name], 'predict'):
                pred = self.models[model_name].predict(X_i)[0]
            else:
                # Handle two-stage model
                pred, _ = self.models[model_name].predict(X_i)
                pred = pred[0]
            
            predictions[i] = pred
        
        return predictions
    
    def update(self, predictions: np.ndarray, actuals: np.ndarray, model_names: List[str]):
        """
        Update model performance statistics
        """
        # Calculate errors
        errors = np.abs(predictions - actuals)
        median_error = np.median(errors)
        
        for i, (error, name) in enumerate(zip(errors, model_names)):
            # Success if error is below median
            if error < median_error:
                self.successes[name] += 1
            else:
                self.failures[name] += 1
        
        # Decay old statistics
        for name in self.models.keys():
            self.successes[name] = 1 + 0.99 * (self.successes[name] - 1)
            self.failures[name] = 1 + 0.99 * (self.failures[name] - 1)


# =============================================================================
# INTEGRATED ADVANCED MODEL
# =============================================================================

class AdvancedPJMPredictor:
    """
    Advanced predictor integrating all components:
    - Two-stage spike prediction
    - QRA for probabilistic forecasts
    - Conformal prediction
    - Dynamic ensemble selection
    """
    
    def __init__(self,
                 peak_spike_threshold: float = 175,
                 offpeak_spike_threshold: float = 100,
                 quantiles: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                 coverage: float = 0.9,
                 n_jobs: int = -1):
        
        self.peak_spike_threshold = peak_spike_threshold
        self.offpeak_spike_threshold = offpeak_spike_threshold
        self.quantiles = quantiles
        self.coverage = coverage
        self.n_jobs = n_jobs
        
        # Initialize components
        self.two_stage_spike = TwoStageSpikePrediction(
            peak_threshold=peak_spike_threshold,
            offpeak_threshold=offpeak_spike_threshold,
            n_jobs=n_jobs
        )
        
        self.qra = QuantileRegressionAveraging(
            quantiles=quantiles,
            n_jobs=n_jobs
        )
        
        self.base_ensemble = None
        self.conformal_predictor = None
        self.thompson_ensemble = None
        
    def _build_base_ensemble(self):
        """Build base ensemble models"""
        
        from sklearn.ensemble import VotingRegressor
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=42
        ,
            tree_method='hist',
            predictor='cpu_predictor'
        )
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=42,
            verbosity=-1
        ,
                device_type='cpu'
            )
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=self.n_jobs,
            random_state=42
        )
        
        return VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ])
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Fit all model components
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        X_cal : pd.DataFrame
            Calibration features (for conformal prediction)
        y_cal : pd.Series
            Calibration targets
        """
        
        print("\n" + "="*70)
        print("TRAINING ADVANCED PJM PREDICTOR")
        print("="*70)
        
        # 1. Train two-stage spike predictor
        print("\n1. Two-Stage Spike Prediction")
        self.two_stage_spike.fit(X_train, y_train)
        
        # 2. Train QRA
        print("\n2. Quantile Regression Averaging")
        self.qra.fit(X_train, y_train)
        
        # 3. Train base ensemble
        print("\n3. Base Ensemble")
        self.base_ensemble = self._build_base_ensemble()
        self.base_ensemble.fit(X_train, y_train)
        
        # 4. Calibrate conformal predictor
        print("\n4. Conformal Prediction Calibration")
        self.conformal_predictor = ConformalPredictor(
            self.base_ensemble,
            coverage=self.coverage
        )
        self.conformal_predictor.calibrate(X_cal, y_cal)
        
        # 5. Initialize Thompson Sampling ensemble
        print("\n5. Thompson Sampling Ensemble")
        self.thompson_ensemble = ThompsonSamplingEnsemble({
            'two_stage': self.two_stage_spike,
            'base_ensemble': self.base_ensemble
        })
        
        print("\nTraining complete!")
        return self
    
    def predict_comprehensive(self, X: pd.DataFrame) -> Dict:
        """
        Generate comprehensive predictions with all methods
        
        Returns:
        --------
        Dictionary containing:
        - 'point_prediction': Point predictions
        - 'spike_probability': Spike probabilities
        - 'quantiles': Quantile predictions
        - 'conformal_lower': Conformal lower bound
        - 'conformal_upper': Conformal upper bound
        - 'thompson_prediction': Thompson sampling prediction
        """
        
        # Two-stage predictions
        two_stage_pred, spike_probs = self.two_stage_spike.predict(X)
        
        # QRA predictions
        quantile_preds = self.qra.predict(X)
        
        # Conformal predictions
        conf_pred, conf_lower, conf_upper = self.conformal_predictor.predict_interval(X)
        
        # Thompson sampling predictions
        thompson_pred = self.thompson_ensemble.predict(X)
        
        return {
            'point_prediction': two_stage_pred,
            'spike_probability': spike_probs,
            'quantiles': quantile_preds,
            'conformal_prediction': conf_pred,
            'conformal_lower': conf_lower,
            'conformal_upper': conf_upper,
            'thompson_prediction': thompson_pred,
            'median_prediction': quantile_preds[0.5]
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive evaluation
        """
        
        # Get all predictions
        preds = self.predict_comprehensive(X_test)
        
        # Basic metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        results = {}
        
        # Evaluate each prediction type
        for pred_name in ['point_prediction', 'median_prediction', 'thompson_prediction']:
            if pred_name in preds:
                mae = mean_absolute_error(y_test, preds[pred_name])
                rmse = np.sqrt(mean_squared_error(y_test, preds[pred_name]))
                results[f'{pred_name}_mae'] = mae
                results[f'{pred_name}_rmse'] = rmse
        
        # Spike detection metrics
        thresholds = self.two_stage_spike._get_dynamic_threshold(X_test)
        actual_spikes = y_test.values > thresholds
        predicted_spikes = preds['spike_probability'] > 0.5
        
        if actual_spikes.sum() > 0:
            tp = (actual_spikes & predicted_spikes).sum()
            results['spike_recall'] = tp / actual_spikes.sum()
            results['spike_precision'] = tp / max(predicted_spikes.sum(), 1)
        
        # Coverage evaluation
        covered = (y_test.values >= preds['conformal_lower']) & (y_test.values <= preds['conformal_upper'])
        results['conformal_coverage'] = covered.mean()
        results['conformal_width'] = (preds['conformal_upper'] - preds['conformal_lower']).mean()
        
        # Quantile evaluation (calibration)
        for q in [0.1, 0.5, 0.9]:
            if q in preds['quantiles']:
                below_q = (y_test.values <= preds['quantiles'][q]).mean()
                results[f'quantile_{q}_calibration'] = below_q
        
        return results


# =============================================================================
# EXAMPLE INTEGRATION WITH YOUR EXISTING MODEL
# =============================================================================

def integrate_advanced_features(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Example of how to integrate these features with your existing model
    """
    
    # Initialize advanced predictor
    advanced_model = AdvancedPJMPredictor(
        peak_spike_threshold=175,
        offpeak_spike_threshold=100,
        quantiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        coverage=0.9,
        n_jobs=-1
    )
    
    # Train model (using validation set for calibration)
    advanced_model.fit(X_train, y_train, X_val, y_val)
    
    # Get comprehensive predictions
    test_predictions = advanced_model.predict_comprehensive(X_test)
    
    # Evaluate
    evaluation_results = advanced_model.evaluate(X_test, y_test)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for metric, value in evaluation_results.items():
        if 'mae' in metric or 'rmse' in metric or 'width' in metric:
            print(f"{metric}: ${value:.2f}")
        else:
            print(f"{metric}: {value:.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'datetime': X_test.index,
        'actual': y_test.values,
        'prediction': test_predictions['point_prediction'],
        'spike_prob': test_predictions['spike_probability'],
        'lower_90': test_predictions['conformal_lower'],
        'upper_90': test_predictions['conformal_upper'],
        'q10': test_predictions['quantiles'][0.1],
        'q50': test_predictions['quantiles'][0.5],
        'q90': test_predictions['quantiles'][0.9]
    })
    
    return advanced_model, results_df, evaluation_results