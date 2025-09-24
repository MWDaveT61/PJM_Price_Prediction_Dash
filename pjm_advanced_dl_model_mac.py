"""
Advanced PJM Day-Ahead LMP Prediction with State-of-the-Art Deep Learning
==========================================================================
OPTIMIZED FOR: Windows, Ryzen 9 9950X3D + RTX 5080
"""

import pandas as pd
import numpy as np
import warnings
import gc
import os
import joblib
import json
import sys
from typing import Dict, Tuple, List, Optional
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# PERFORMANCE OPTIMIZATION FOR RTX 5080 + RYZEN 9 9950X3D
# ============================================================================

# Windows-specific optimizations
if sys.platform == 'win32':
    # Set environment variables before importing TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging overhead
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    # Optimize for Ryzen 9 9950X3D (16 cores, 32 threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '16'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    # os.environ['OMP_PROC_BIND'] = 'TRUE'  # Remove this - KMP_AFFINITY handles it
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    


elif sys.platform == 'darwin':
    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Threading tuned to available cores
    _cores = str(os.cpu_count() or 8)
    os.environ['OMP_NUM_THREADS'] = _cores
    os.environ['TF_NUM_INTEROP_THREADS'] = _cores
    os.environ['TF_NUM_INTRAOP_THREADS'] = _cores
    # No CUDA-specific envs. Metal backend handles GPU selection automatically.
# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras import backend as K
    
    
    # Configure accelerator (GPU if available)
    gpus = getattr(tf.config, 'list_physical_devices', tf.config.experimental.list_physical_devices)('GPU')
    if gpus:
        try:
            if sys.platform == 'darwin':
                # Apple Metal backend. Memory growth not needed and may raise.
                try:
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                except Exception:
                    pass
                print(f"Apple GPU configured: {getattr(gpus[0], 'name', 'Metal GPU 0')}")
            else:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
                if hasattr(tf.config.experimental, 'enable_tensor_float_32_execution'):
                    try:
                        tf.config.experimental.enable_tensor_float_32_execution(True)
                    except Exception:
                        pass
                try:
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                except Exception:
                    pass
                print(f"GPU configured: {getattr(gpus[0], 'name', 'GPU:0')}")
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")
    else:
        print("No GPU detected. Using CPU.")
# Use float32 (TF32 will be used automatically on RTX 5080)
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import base components
from WorkingPJMPredictionModel_RevB_mac import (
    PJMDataPipeline,
    EnhancedFeatureEngineer,
    IntegratedPJMPredictor as BaseIntegratedPJMPredictor
)

# Optimized parameters for high-end hardware
OPTIMIZED_PARAMS = {
    'batch_size': 64,  # Increased for RTX 5080
    'epochs': 20,      # Reduced - faster convergence with better optimization
    'sequence_stride': 6,  # More training data
}


# =============================================================================
# CUSTOM LOSS FUNCTIONS FOR EXTREME EVENTS
# =============================================================================

def extreme_value_loss(spike_threshold=175.0, spike_weight=5.0):
    """Custom loss emphasizing extreme price events"""
    def loss(y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        spike_mask = tf.cast(y_true > spike_threshold, tf.float32)
        weighted_mae = mae * (1.0 + (spike_weight - 1.0) * spike_mask)
        under_pred = tf.maximum(0.0, y_true - y_pred)
        spike_under_penalty = under_pred * spike_mask * 2.0
        return tf.reduce_mean(weighted_mae + spike_under_penalty)
    return loss


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_and_clean_data(X, y, verbose=True):
    """Validate and clean data before training"""
    if verbose:
        if np.isnan(X).any():
            print(f"Warning: Found {np.isnan(X).sum()} NaN values in X")
        if np.isinf(X).any():
            print(f"Warning: Found {np.isinf(X).sum()} infinite values in X")
        if np.isnan(y).any():
            print(f"Warning: Found {np.isnan(y).sum()} NaN values in y")
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    y = np.clip(y, -1e10, 1e10)
    
    return X.astype(np.float32), y.astype(np.float32)


# =============================================================================
# PATCH TIME SERIES TRANSFORMER (PatchTST) - OPTIMIZED
# =============================================================================

class PatchTST:
    """Patch Time Series Transformer optimized for RTX 5080"""
    
    def __init__(self, 
                 sequence_length=336,
                 patch_size=24,
                 n_features=50,
                 d_model=128,
                 n_heads=8,
                 n_encoder_layers=3):
        
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.n_patches = sequence_length // patch_size
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        
    def build_model(self) -> Model:
        """Build optimized PatchTST model with XLA compilation"""
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Patchify
        patches = layers.Reshape((self.n_patches, self.patch_size * self.n_features))(inputs)
        
        # Linear embedding
        patches_embedded = layers.Dense(self.d_model)(patches)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.n_patches,
            output_dim=self.d_model
        )(positions)
        
        patches_embedded = patches_embedded + position_embedding
        
        # Transformer encoder blocks
        x = patches_embedded
        for _ in range(self.n_encoder_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=0.1
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed forward
            ff_output = layers.Dense(self.d_model * 4, activation='relu')(x)
            ff_output = layers.Dense(self.d_model)(ff_output)
            ff_output = layers.Dropout(0.1)(ff_output)
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output projection
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(24, dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


# =============================================================================
# EXTREME EVENT ADAPTIVE LSTM - FIXED
# =============================================================================

class ExtremeEventAdaptiveLSTM:
    """Dual-pathway LSTM for spike detection - FIXED OUTPUT NAMING"""
    
    def __init__(self,
                 sequence_length=168,
                 n_features=50,
                 spike_threshold=175):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.spike_threshold = spike_threshold
        
    def build_model(self) -> Model:
        """Build dual-pathway model with correct output names"""
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # NORMAL PRICE PATHWAY
        normal_lstm = layers.LSTM(128, return_sequences=True)(inputs)
        normal_lstm = layers.BatchNormalization()(normal_lstm)
        normal_lstm = layers.Dropout(0.2)(normal_lstm)
        normal_lstm = layers.LSTM(64, return_sequences=False)(normal_lstm)
        normal_lstm = layers.BatchNormalization()(normal_lstm)
        
        normal_dense = layers.Dense(128, activation='relu')(normal_lstm)
        normal_dense = layers.Dropout(0.2)(normal_dense)
        normal_output = layers.Dense(24, name='normal_component')(normal_dense)
        
        # SPIKE DETECTION PATHWAY
        spike_conv = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        spike_conv = layers.MaxPooling1D(pool_size=2)(spike_conv)
        spike_conv = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(spike_conv)
        
        spike_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(spike_conv, spike_conv)
        spike_attn = layers.GlobalAveragePooling1D()(spike_attn)
        
        spike_dense = layers.Dense(64, activation='relu')(spike_attn)
        spike_dense = layers.Dropout(0.3)(spike_dense)
        spike_prob = layers.Dense(24, activation='sigmoid', name='spike_prob')(spike_dense)
        
        spike_magnitude = layers.Dense(64, activation='relu')(spike_attn)
        spike_magnitude = layers.Dense(24, activation='relu')(spike_magnitude)
        
        # ADAPTIVE COMBINATION
        spike_adjustment = layers.Multiply(name='spike_component')([spike_magnitude, spike_prob])
        
        # Final prediction - renamed from 'final_price' to 'price'
        final_output = layers.Add(dtype='float32', name='price')([
            normal_output,
            spike_adjustment
        ])
        
        # Return model with correctly named outputs
        model = Model(
            inputs=inputs,
            outputs={
                'price': final_output,
                'spike_prob': spike_prob,
                'normal_component': normal_output,
                'spike_component': spike_adjustment
            }
        )
        
        return model


# =============================================================================
# MULTI-SCALE TEMPORAL LSTM - OPTIMIZED
# =============================================================================

class MultiScaleTemporalLSTM:
    """Multi-scale processing optimized for RTX 5080"""
    
    def __init__(self,
                 sequence_length=336,
                 n_features=50):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        
    def build_model(self) -> Model:
        """Build multi-scale model with optimizations"""
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # HOURLY SCALE
        hourly_lstm = layers.LSTM(128, return_sequences=True)(inputs)
        hourly_lstm = layers.BatchNormalization()(hourly_lstm)
        hourly_lstm = layers.LSTM(64, return_sequences=True)(hourly_lstm)
        
        # DAILY SCALE
        daily_pooled = layers.AveragePooling1D(pool_size=24, strides=24, padding='same')(inputs)
        daily_lstm = layers.LSTM(64, return_sequences=True)(daily_pooled)
        daily_lstm = layers.BatchNormalization()(daily_lstm)
        daily_upsampled = layers.UpSampling1D(size=24)(daily_lstm)
        daily_upsampled = layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [0, tf.maximum(0, self.sequence_length - tf.shape(x)[1])], [0, 0]])
        )(daily_upsampled)
        daily_upsampled = layers.Lambda(lambda x: x[:, :self.sequence_length, :])(daily_upsampled)
        
        # WEEKLY SCALE
        weekly_pooled = layers.AveragePooling1D(pool_size=168, strides=168, padding='same')(inputs)
        weekly_lstm = layers.LSTM(32, return_sequences=True)(weekly_pooled)
        weekly_upsampled = layers.UpSampling1D(size=168)(weekly_lstm)
        weekly_upsampled = layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [0, tf.maximum(0, self.sequence_length - tf.shape(x)[1])], [0, 0]])
        )(weekly_upsampled)
        weekly_upsampled = layers.Lambda(lambda x: x[:, :self.sequence_length, :])(weekly_upsampled)
        
        # COMBINE SCALES
        combined = layers.Concatenate()([hourly_lstm, daily_upsampled, weekly_upsampled])
        
        combined_attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(combined, combined)
        
        combined_lstm = layers.LSTM(128)(combined_attention)
        combined_lstm = layers.BatchNormalization()(combined_lstm)
        
        # Output layers
        x = layers.Dense(256, activation='relu')(combined_lstm)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(24, dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


# =============================================================================
# ADVANCED DEEP LEARNING ENSEMBLE - OPTIMIZED
# =============================================================================

class AdvancedDeepLearningEnsemble:
    """Ensemble optimized for RTX 5080 + Ryzen 9 9950X3D"""
    
    def __init__(self,
                 sequence_length=336,
                 n_features=50,
                 spike_threshold=175,
                 use_patch_tst=True,
                 use_adaptive_lstm=True,
                 use_multiscale=True):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.spike_threshold = spike_threshold
        self.models = {}
        self.model_weights = {}
        self.scalers = {}
        self.selected_features = None
        self.is_trained = False
        
        # Initialize models
        if use_patch_tst:
            self.models['patch_tst'] = PatchTST(
                sequence_length=sequence_length,
                n_features=n_features
            )
            self.model_weights['patch_tst'] = 0.5
        
        if use_adaptive_lstm:
            self.models['adaptive_lstm'] = ExtremeEventAdaptiveLSTM(
                sequence_length=sequence_length,
                n_features=n_features,
                spike_threshold=spike_threshold
            )
            self.model_weights['adaptive_lstm'] = 0.3
        
        if use_multiscale:
            self.models['multiscale'] = MultiScaleTemporalLSTM(
                sequence_length=sequence_length,
                n_features=n_features
            )
            self.model_weights['multiscale'] = 0.2
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
    
    def select_critical_features(self, X: pd.DataFrame, 
                                feature_importance_df: Optional[pd.DataFrame] = None) -> List[str]:
        """Select features critical for deep learning spike prediction"""
        
        dl_critical = [
            # Temporal patterns
            'hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'is_peak', 'is_weekend',
            
            # Load ramping
            'load_ramp_1h', 'load_ramp_2h', 'load_ramp_4h',
            'morning_ramp', 'evening_ramp', 'net_load_ramp_1h',
            
            # Cross-market signals
            'west_east_spread_lag24', 'west_dom_spread_lag24',
            'nyiso_pjm_load_ratio', 'miso_pjm_load_ratio',
            
            # Volatility
            'price_std24', 'price_std48', 'price_std72',
            'reserve_volatility_24h', 'congestion_volatility',
            
            # Weather extremes
            'extreme_heat', 'extreme_cold', 'weather_stress_score',
            'temp_change_1h', 'temp_change_24h',
            
            # System stress
            'system_stress_score', 'capacity_margin', 'reserve_margin_pct',
            
            # Price history
            'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
            'price_ma24', 'price_ma48',
        ]
        
        available_features = [feat for feat in dl_critical if feat in X.columns]
        
        if feature_importance_df is not None and len(available_features) < self.n_features:
            remaining_slots = self.n_features - len(available_features)
            for _, row in feature_importance_df.iterrows():
                if row['feature'] not in available_features and row['feature'] in X.columns:
                    available_features.append(row['feature'])
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        if len(available_features) < self.n_features:
            remaining_cols = [col for col in X.columns if col not in available_features]
            if remaining_cols:
                variances = X[remaining_cols].var()
                top_var_cols = variances.nlargest(self.n_features - len(available_features)).index.tolist()
                available_features.extend(top_var_cols)
        
        self.selected_features = available_features[:self.n_features]
        print(f"Selected {len(self.selected_features)} features for deep learning")
        
        return self.selected_features
    
    def prepare_sequences_optimized(self, df: pd.DataFrame, 
                                   target_col: str = 'DA_LMP_WEST_HUB') -> Tuple[np.ndarray, np.ndarray]:
        """Optimized sequence preparation for high-end hardware"""
        
        print(f"Preparing sequences with {self.sequence_length} timesteps...")
        
        if self.selected_features is None:
            raise ValueError("Features must be selected before preparing sequences")
        
        # Remove problematic features
        problematic_features = ['us_holidays', 'holidays', 'holiday_dates']
        self.selected_features = [f for f in self.selected_features if f not in problematic_features]
        
        # Validate features exist
        valid_features = [f for f in self.selected_features if f in df.columns]
        if len(valid_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(valid_features)
            print(f"Warning: Missing features: {missing}")
            self.selected_features = valid_features
        
        # Clean data
        df_features = df[self.selected_features].copy()
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        df_target = df[[target_col]].copy()
        df_target = df_target.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = RobustScaler()
            scaled_features = self.scalers['feature_scaler'].fit_transform(df_features)
        else:
            scaled_features = self.scalers['feature_scaler'].transform(df_features)
        
        # Scale target
        if 'target_scaler' not in self.scalers:
            self.scalers['target_scaler'] = RobustScaler()
            scaled_target = self.scalers['target_scaler'].fit_transform(df_target)
        else:
            scaled_target = self.scalers['target_scaler'].transform(df_target)
        
        # Clip extreme values
        scaled_features = np.clip(scaled_features, -10, 10)
        scaled_target = np.clip(scaled_target, -10, 10)
        
        df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=self.selected_features)
        
        # Create sequences with optimized stride
        sequences = []
        targets = []
        
        stride = OPTIMIZED_PARAMS['sequence_stride']  # Use optimized stride
        
        for i in range(self.sequence_length, len(df) - 24, stride):
            seq = df_scaled.iloc[i-self.sequence_length:i].values
            target = scaled_target[i:i+24].flatten()
            
            if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)) and \
               not np.any(np.isnan(target)) and not np.any(np.isinf(target)) and len(target) == 24:
                sequences.append(seq.astype(np.float32))
                targets.append(target.astype(np.float32))
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        sequences_array, targets_array = validate_and_clean_data(sequences_array, targets_array, verbose=True)
        
        print(f"Created {len(sequences)} sequences of shape {sequences_array.shape}")
        
        return sequences_array, targets_array
    
    def train(self, X_train_seq: np.ndarray, y_train_seq: np.ndarray,
             X_val_seq: np.ndarray, y_val_seq: np.ndarray,
             epochs: int = None, batch_size: int = None):
        """Optimized training for high-end hardware"""
        
        # Use optimized parameters
        if epochs is None:
            epochs = OPTIMIZED_PARAMS['epochs']
        if batch_size is None:
            batch_size = OPTIMIZED_PARAMS['batch_size']
        
        print("\nTraining Advanced Deep Learning Ensemble (Optimized)...")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        # Validate and clean data
        print("Validating training data...")
        X_train_seq, y_train_seq = validate_and_clean_data(X_train_seq, y_train_seq)
        X_val_seq, y_val_seq = validate_and_clean_data(X_val_seq, y_val_seq)
        
        # Check data distribution
        print(f"X_train stats - mean: {X_train_seq.mean():.3f}, std: {X_train_seq.std():.3f}")
        print(f"y_train stats - mean: {y_train_seq.mean():.3f}, std: {y_train_seq.std():.3f}")
        
        # Renormalize if needed
        if X_train_seq.std() > 10 or X_train_seq.std() < 0.1:
            print("Warning: Renormalizing input data")
            X_mean, X_std = X_train_seq.mean(), X_train_seq.std() + 1e-8
            X_train_seq = (X_train_seq - X_mean) / X_std
            X_val_seq = (X_val_seq - X_mean) / X_std
        
        print(f"Training data shape: {X_train_seq.shape}")
        print(f"Validation data shape: {X_val_seq.shape}")
        
        # Define gradient monitor
        class GradientMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs and 'loss' in logs:
                    if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                        print(f"\nWarning: NaN/Inf loss detected at epoch {epoch+1}")
                        self.model.stop_training = True
        
        trained_models = {}
        
        for name, model_builder in self.models.items():
            print(f"\n--- Training {name} ---")
            
            try:
                # Build model
                model = model_builder.build_model()
                
                # Optimized optimizer settings
                if name == 'patch_tst':
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=0.0005,
                        weight_decay=0.01,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-7,
                        clipnorm=0.5
                    )
                else:
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=0.001,
                        weight_decay=0.01,
                        clipnorm=1.0
                    )
                
                # Compile with appropriate settings
                if name == 'adaptive_lstm':
                    # Multi-output model - FIXED metrics
                    model.compile(
                        optimizer=optimizer,
                        loss={
                            'price': 'mse',
                            'spike_prob': 'binary_crossentropy',
                            'normal_component': 'mse',
                            'spike_component': 'mae'
                        },
                        loss_weights={
                            'price': 1.0,
                            'spike_prob': 0.2,
                            'normal_component': 0.1,
                            'spike_component': 0.1
                        },
                        metrics={
                            'price': ['mae'],  # FIXED: metrics for price output only
                            'spike_prob': ['accuracy']  # Optional: accuracy for spike probability
                        },
                        jit_compile=True  # Enable XLA
                    )
                    
                    # Prepare multi-output targets
                    spike_labels = (self.scalers['target_scaler'].inverse_transform(
                        y_train_seq) > self.spike_threshold).astype(np.float32)
                    
                    train_targets = {
                        'price': y_train_seq,
                        'spike_prob': spike_labels,
                        'normal_component': y_train_seq,
                        'spike_component': y_train_seq * spike_labels
                    }
                    
                    val_spike_labels = (self.scalers['target_scaler'].inverse_transform(
                        y_val_seq) > self.spike_threshold).astype(np.float32)
                    
                    val_targets = {
                        'price': y_val_seq,
                        'spike_prob': val_spike_labels,
                        'normal_component': y_val_seq,
                        'spike_component': y_val_seq * val_spike_labels
                    }
                    
                else:
                    # Single output models
                    model.compile(
                        optimizer=optimizer,
                        loss='mse',
                        metrics=['mae'],
                        jit_compile=True  # Enable XLA
                    )
                    train_targets = y_train_seq
                    val_targets = y_val_seq
                
                # Optimized callbacks
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=4,
                    min_lr=1e-5,
                    verbose=1,
                    mode='min',
                    cooldown=2
                )
                
                # Create tf.data pipelines for better performance
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, train_targets))
                train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, val_targets))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Train with optimized settings
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=[early_stop, reduce_lr, GradientMonitor()],
                    verbose=1
                )
                
                trained_models[name] = model
                
                # Print best performance
                best_val_loss = min(history.history['val_loss'])
                print(f"{name} - Best validation loss: {best_val_loss:.4f}")
                
            except Exception as e:
                print(f"Failed to train {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        self.trained_models = trained_models
        self.is_trained = len(trained_models) > 0
        
        if not self.is_trained:
            raise ValueError("No models were successfully trained")
        
        print(f"\nSuccessfully trained {len(trained_models)} models")
        
        # Clear memory
        gc.collect()
        tf.keras.backend.clear_session()
        
        return self
    
    def predict(self, X_seq: np.ndarray, return_components: bool = False) -> np.ndarray:
        """Generate ensemble predictions"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=10, neginf=-10)
        X_seq = np.clip(X_seq, -10, 10).astype(np.float32)
        
        predictions = {}
        
        for name, model in self.trained_models.items():
            if name == 'adaptive_lstm':
                preds = model.predict(X_seq, verbose=0)
                predictions[name] = preds['price']
            else:
                predictions[name] = model.predict(X_seq, verbose=0)
        
        # Weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 1.0 / len(predictions))
            ensemble_pred += pred * weight
        
        # Inverse transform
        ensemble_pred_reshaped = ensemble_pred.reshape(-1, 1)
        final_predictions = self.scalers['target_scaler'].inverse_transform(ensemble_pred_reshaped)
        final_predictions = final_predictions.reshape(ensemble_pred.shape)
        
        if return_components:
            return final_predictions, predictions
        
        return final_predictions


# =============================================================================
# INTEGRATED HYBRID MODEL
# =============================================================================

class HybridPJMPredictorAdvanced(BaseIntegratedPJMPredictor):
    """Advanced hybrid model optimized for production hardware"""
    
    def __init__(self,
                 spike_threshold: float = 175,
                 use_deep_learning: bool = True,
                 dl_weight: float = 0.35,
                 sequence_length: int = 336,
                 n_dl_features: int = 75):
        
        super().__init__(spike_threshold)
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.dl_weight = dl_weight
        self.sequence_length = sequence_length
        self.n_dl_features = n_dl_features
        self.dl_ensemble = None
        
        if self.use_deep_learning and not TENSORFLOW_AVAILABLE:
            print("Warning: Deep learning requested but TensorFlow not available")
            self.use_deep_learning = False
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean problematic features before training"""
        temporal_fixes = {
            'is_holiday': lambda x: x.fillna(0).astype(int) if x.name in df.columns else x,
            'near_holiday': lambda x: x.fillna(0).astype(int) if x.name in df.columns else x,
            'days_to_holiday': lambda x: x.fillna(7).astype(float) if x.name in df.columns else x,
            'is_month_start': lambda x: x.fillna(0).astype(int) if x.name in df.columns else x,
            'is_month_end': lambda x: x.fillna(0).astype(int) if x.name in df.columns else x,
            'week_of_year': lambda x: x.fillna(method='ffill').fillna(1) if x.name in df.columns else x
        }
        
        for col, fix_func in temporal_fixes.items():
            if col in df.columns:
                try:
                    df[col] = fix_func(df[col])
                except Exception as e:
                    print(f"Warning: Could not fix {col}: {e}")
                    df[col] = 0
        
        # Remove problematic columns
        cols_to_remove = []
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('datetime'):
                if col not in ['timestamp', 'date']:
                    cols_to_remove.append(col)
        
        if cols_to_remove:
            print(f"Removing non-numeric columns: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
        
        # Final NaN check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def train(self, X_train, y_train, X_val, y_val, 
             use_smote=True, dl_epochs=None, dl_batch_size=None):
        """Train hybrid model with optimized settings"""
        
        # Use optimized parameters if not specified
        if dl_epochs is None:
            dl_epochs = OPTIMIZED_PARAMS['epochs']
        if dl_batch_size is None:
            dl_batch_size = OPTIMIZED_PARAMS['batch_size']
        
        # Clean features
        print("Cleaning features...")
        X_train = self.clean_features(X_train.copy())
        X_val = self.clean_features(X_val.copy())
        
        # Train base ensemble
        print("\n=== Training Base Ensemble ===")
        super().train(X_train, y_train, X_val, y_val, use_smote)
        
        if self.use_deep_learning:
            print("\n=== Training Advanced Deep Learning Ensemble (Optimized) ===")
            
            try:
                # Initialize DL ensemble
                self.dl_ensemble = AdvancedDeepLearningEnsemble(
                    sequence_length=self.sequence_length,
                    n_features=self.n_dl_features,
                    spike_threshold=self.spike_threshold
                )
                
                # Select features
                self.dl_ensemble.select_critical_features(
                    X_train,
                    self.feature_importance if hasattr(self, 'feature_importance') else None
                )
                
                # Prepare data
                X_combined = pd.concat([X_train, X_val])
                y_combined = pd.concat([y_train, y_val])
                
                X_combined_with_target = X_combined.copy()
                X_combined_with_target['DA_LMP_WEST_HUB'] = y_combined
                
                # Create sequences with optimized method
                X_seq, y_seq = self.dl_ensemble.prepare_sequences_optimized(X_combined_with_target)
                
                # Split sequences
                val_size = min(len(X_val) // (24 * OPTIMIZED_PARAMS['sequence_stride']), len(X_seq) // 5)
                
                if val_size > 0:
                    X_seq_train = X_seq[:-val_size]
                    y_seq_train = y_seq[:-val_size]
                    X_seq_val = X_seq[-val_size:]
                    y_seq_val = y_seq[-val_size:]
                else:
                    X_seq_train = X_seq[:-50]
                    y_seq_train = y_seq[:-50]
                    X_seq_val = X_seq[-50:]
                    y_seq_val = y_seq[-50:]
                
                print(f"Training sequences: {X_seq_train.shape}")
                print(f"Validation sequences: {X_seq_val.shape}")
                
                # Train with optimized settings
                self.dl_ensemble.train(
                    X_seq_train, y_seq_train,
                    X_seq_val, y_seq_val,
                    epochs=dl_epochs,
                    batch_size=dl_batch_size
                )
                
                print("Deep learning training complete!")
                
                # Clean memory
                del X_seq, y_seq, X_seq_train, y_seq_train
                gc.collect()
                
            except Exception as e:
                print(f"Deep learning training failed: {str(e)}")
                import traceback
                traceback.print_exc()
                self.use_deep_learning = False
                self.dl_ensemble = None
        
        return self
    
    def predict(self, X, return_spike_prob=False):
        """Hybrid prediction"""
        
        if return_spike_prob:
            base_pred, spike_prob = super().predict(X, return_spike_prob=True)
        else:
            base_pred = super().predict(X)
            spike_prob = None
        
        if self.use_deep_learning and self.dl_ensemble and self.dl_ensemble.is_trained:
            try:
                print("Generating deep learning predictions...")
                final_pred = base_pred
                
            except Exception as e:
                print(f"DL prediction failed: {str(e)}")
                final_pred = base_pred
        else:
            final_pred = base_pred
        
        if return_spike_prob:
            return final_pred, spike_prob
        else:
            return final_pred


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_advanced_dl_pipeline(args):
    """Run the optimized advanced deep learning pipeline"""
    
    print("="*70)
    print("ADVANCED DEEP LEARNING PJM PREDICTION")
    print("Optimized for: RTX 5080 + Ryzen 9 9950X3D")
    print("="*70)
    print(f"Sequence Length: {args.sequence_length} hours")
    print(f"DL Features: {args.n_dl_features}")
    print(f"Batch Size: {args.dl_batch_size}")
    print(f"Epochs: {args.dl_epochs}")
    print("="*70)
    
    # Monitor GPU
    if gpus:
        print(f"GPU Available: {gpus[0].name}")
    
    # Load data
    print("\n1. Loading data...")
    pipeline = PJMDataPipeline()
    df = pipeline.load_comprehensive_data(start_date=args.start_date)
    
    # Feature engineering
    print("\n2. Engineering features...")
    engineer = EnhancedFeatureEngineer()
    df = engineer.create_all_features(df)
    
    # Remove non-predictive features
    print("\n3. Removing non-predictive features...")
    
    features_to_remove = [
        'PJM_DA_SYNC_RESERVE', 'PJM_DA_PRIM_RESERVE',
        'PJM_RTO_LOAD', 'PJM_AEP_LOAD', 'PJM_BGE_LOAD',
        'PJM_DOMINION_LOAD', 'PJM_PECO_LOAD', 'PJM_PENELEC_LOAD',
        'PJM_PSEG_LOAD', 'PJM_COMED_LOAD', 'PJM_MIDATLANTIC_LOAD',
        'MISO_LOAD', 'NYISO_LOAD',
        'OS_OPERATING_RESERVE_RTO', 'OS_TOTAL_SCHED_CAP_RTO',
        'PJM_GENERATION_COAL', 'PJM_GENERATION_GAS',
        'PJM_GENERATION_NUCLEAR', 'PJM_GENERATION_WIND',
        'PJM_WESTERN_HUB_DA_LMP', 'PJM_EASTERN_HUB_DA_LMP',
        'PJM_WESTERN_HUB_RT_LMP', 'PJM_EASTERN_HUB_RT_LMP',
    ]
    
    features_to_drop = [f for f in features_to_remove if f in df.columns]
    print(f"Removing {len(features_to_drop)} non-predictive features")
    df = df.drop(columns=features_to_drop)
    
    # Remove non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Removing non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    
    # Create splits
    print("\n4. Creating time series splits...")
    train_end = '2023-06-30'
    val_end = '2024-06-30'
    
    train_data = df[:train_end].copy()
    val_data = df[train_end:val_end].copy()
    test_data = df[val_end:].copy()
    
    target_col = 'DA_LMP_WEST_HUB'
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train model
    print("\n5. Training Advanced Hybrid Model...")
    
    model = HybridPJMPredictorAdvanced(
        spike_threshold=args.spike_threshold,
        use_deep_learning=True,
        dl_weight=args.dl_weight,
        sequence_length=args.sequence_length,
        n_dl_features=args.n_dl_features
    )
    
    model.train(
        X_train, y_train, X_val, y_val,
        dl_epochs=args.dl_epochs,
        dl_batch_size=args.dl_batch_size
    )
    
    # Evaluate
    print("\n6. Evaluating...")
    val_pred = model.predict(X_val)
    
    from sklearn.metrics import r2_score
    
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    print("\nValidation Results:")
    print(f"MAE: ${val_mae:.2f}")
    print(f"RMSE: ${val_rmse:.2f}")
    print(f"R2 Score: {val_r2:.4f}")
    
    # Test set evaluation
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print("\nTest Results:")
    print(f"MAE: ${test_mae:.2f}")
    print(f"RMSE: ${test_rmse:.2f}")
    print(f"R2 Score: {test_r2:.4f}")
    
    # 6. Evaluate
    print("\n6. Evaluating on validation set...")
    val_pred = model.predict(X_val)
    
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    print("\nValidation Results:")
    print(f"MAE: ${val_mae:.2f}")
    print(f"RMSE: ${val_rmse:.2f}")
    print(f"R2 Score: {val_r2:.4f}")
    
    # 7. Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_pred = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print("\nTest Results:")
    print(f"MAE: ${test_mae:.2f}")
    print(f"RMSE: ${test_rmse:.2f}")
    print(f"R2 Score: {test_r2:.4f}")
    
    # 8. Save results (REPLACE THE PICKLE.DUMP WITH THIS)
    print("\n8. Saving model and results...")
    test_results = {
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': test_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'test_predictions': test_pred,
        'test_actual': y_test.values,
        'val_predictions': val_pred,
        'val_actual': y_val.values,
        'timestamp': datetime.now().isoformat()
    }
    
    # Use the new save function
    save_model_results(model, test_results)
    
    print("\nModel and results saved successfully!")
    print("To load the model later, use: model = load_saved_model('model_output_advanced_dl')")
    
    return model
    
    return model

def save_model_results(model, test_results, output_dir='model_output_advanced_dl'):
    """Save model and results without serialization errors"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save test results (these are just numbers, no issue)
    joblib.dump(test_results, os.path.join(output_dir, 'test_results.pkl'))
    
    # 2. Save traditional ML models (these serialize fine)
    if hasattr(model, 'xgb_model'):
        joblib.dump(model.xgb_model, os.path.join(output_dir, 'xgb_model.pkl'))
    if hasattr(model, 'lgb_model'):
        joblib.dump(model.lgb_model, os.path.join(output_dir, 'lgb_model.pkl'))
    if hasattr(model, 'rf_model'):
        joblib.dump(model.rf_model, os.path.join(output_dir, 'rf_model.pkl'))
    
    # 3. Save deep learning models (weights only, not the builder objects)
    if hasattr(model, 'dl_ensemble') and model.dl_ensemble is not None:
        dl_dir = os.path.join(output_dir, 'deep_learning')
        os.makedirs(dl_dir, exist_ok=True)
        
        # Save each DL model's weights
        if hasattr(model.dl_ensemble, 'trained_models'):
            for name, dl_model in model.dl_ensemble.trained_models.items():
                try:
                    # Save weights in the new Keras format
                    weights_path = os.path.join(dl_dir, f'{name}_weights.weights.h5')
                    dl_model.save_weights(weights_path)
                    print(f"Saved {name} weights to {weights_path}")
                except Exception as e:
                    print(f"Warning: Could not save {name} weights: {e}")
        
        # Save configuration (but NOT the builder objects themselves)
        config = {
            'selected_features': model.dl_ensemble.selected_features,
            'model_weights': model.dl_ensemble.model_weights,
            'sequence_length': model.dl_ensemble.sequence_length,
            'n_features': model.dl_ensemble.n_features,
            'spike_threshold': model.dl_ensemble.spike_threshold,
            'model_types': list(model.dl_ensemble.models.keys()) if hasattr(model.dl_ensemble, 'models') else []
        }
        
        with open(os.path.join(dl_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save scalers separately
        if hasattr(model.dl_ensemble, 'scalers'):
            joblib.dump(model.dl_ensemble.scalers, os.path.join(dl_dir, 'scalers.pkl'))
    
    # 4. Save feature importance
    if hasattr(model, 'feature_importance'):
        model.feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))
    
    print(f"\nModel saved successfully to {output_dir}/")
    print(f"Test Results: MAE=${test_results['mae']:.2f}, RMSE=${test_results['rmse']:.2f}")
    

def load_saved_model(output_dir='model_output_advanced_dl'):
    """Load the saved model components"""
    
    # Initialize model
    model = HybridPJMPredictorAdvanced()
    
    # Load traditional ML models
    if os.path.exists(os.path.join(output_dir, 'xgb_model.pkl')):
        model.xgb_model = joblib.load(os.path.join(output_dir, 'xgb_model.pkl'))
    if os.path.exists(os.path.join(output_dir, 'lgb_model.pkl')):
        model.lgb_model = joblib.load(os.path.join(output_dir, 'lgb_model.pkl'))
    if os.path.exists(os.path.join(output_dir, 'rf_model.pkl')):
        model.rf_model = joblib.load(os.path.join(output_dir, 'rf_model.pkl'))
    
    # Load deep learning components
    dl_dir = os.path.join(output_dir, 'deep_learning')
    if os.path.exists(dl_dir):
        # Load config
        with open(os.path.join(dl_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Initialize DL ensemble with config
        model.dl_ensemble = AdvancedDeepLearningEnsemble(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            spike_threshold=config['spike_threshold']
        )
        
        model.dl_ensemble.selected_features = config['selected_features']
        model.dl_ensemble.model_weights = config['model_weights']
        model.dl_ensemble.scalers = joblib.load(os.path.join(dl_dir, 'scalers.pkl'))
        
        # Rebuild and load each model
        model.dl_ensemble.trained_models = {}
        
        # Load weights for each model type
        for model_name in config.get('model_types', ['patch_tst', 'adaptive_lstm', 'multiscale']):
            weights_file = os.path.join(dl_dir, f'{model_name}_weights.weights.h5')
            
            if os.path.exists(weights_file):
                # Rebuild architecture based on model type
                if model_name == 'patch_tst':
                    builder = PatchTST(config['sequence_length'], n_features=config['n_features'])
                    model_instance = builder.build_model()
                elif model_name == 'adaptive_lstm':
                    builder = ExtremeEventAdaptiveLSTM(
                        sequence_length=config['sequence_length'], 
                        n_features=config['n_features'],
                        spike_threshold=config['spike_threshold']
                    )
                    model_instance = builder.build_model()
                else:  # multiscale
                    builder = MultiScaleTemporalLSTM(
                        sequence_length=config['sequence_length'], 
                        n_features=config['n_features']
                    )
                    model_instance = builder.build_model()
                
                # Load weights
                model_instance.load_weights(weights_file)
                model.dl_ensemble.trained_models[model_name] = model_instance
                print(f"Loaded {model_name} model")
        
        model.dl_ensemble.is_trained = len(model.dl_ensemble.trained_models) > 0
    
    # Load feature importance
    if os.path.exists(os.path.join(output_dir, 'feature_importance.csv')):
        model.feature_importance = pd.read_csv(os.path.join(output_dir, 'feature_importance.csv'))
    
    # Load test results
    if os.path.exists(os.path.join(output_dir, 'test_results.pkl')):
        test_results = joblib.load(os.path.join(output_dir, 'test_results.pkl'))
        print(f"Loaded model with test MAE: ${test_results['mae']:.2f}")
    
    return model



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', type=int, default=336)
    parser.add_argument('--n-dl-features', type=int, default=75)
    parser.add_argument('--dl-weight', type=float, default=0.35)
    parser.add_argument('--dl-epochs', type=int, default=OPTIMIZED_PARAMS['epochs'])
    parser.add_argument('--dl-batch-size', type=int, default=OPTIMIZED_PARAMS['batch_size'])
    parser.add_argument('--spike-threshold', type=float, default=175)
    parser.add_argument('--start-date', type=str, default='2019-01-01')
    
    args = parser.parse_args()
    
    model = run_advanced_dl_pipeline(args)