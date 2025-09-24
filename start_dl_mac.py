from pjm_advanced_dl_model_mac import HybridPJMPredictorAdvanced
from WorkingPJMPredictionModel_RevB_mac import PJMDataPipeline, EnhancedFeatureEngineer
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_absolute_error
import os
import pickle
from datetime import datetime

# Create arguments for advanced model
args = argparse.Namespace(
    # Advanced DL settings
    use_deep_learning=True,
    sequence_length=336,
    n_dl_features=75,
    dl_weight=0.35,
    dl_epochs=30,
    dl_batch_size=16,
    start_date='2018-01-01',    
    test_days=75,               
    spike_threshold=125,        
    output_dir='model_output_advanced_dl',
    verbose=True
)

print("="*70)
print("ADVANCED DEEP LEARNING PJM PREDICTION (macOS/M4 Max)")
print("="*70)

# 1. Load data
print("\n1. Loading data from SQL Server...")
pipeline = PJMDataPipeline()
df = pipeline.load_comprehensive_data(start_date=args.start_date)

# 2. Feature engineering
print("\n2. Engineering features...")
engineer = EnhancedFeatureEngineer()
df = engineer.create_all_features(df)

# Check what columns we have after feature engineering
print("\n   Checking for target column...")
if 'DA_LMP_WEST_HUB' not in df.columns:
    # Try to find the actual column name
    da_lmp_cols = [col for col in df.columns if 'DA_LMP' in col and 'WEST' in col]
    
    if da_lmp_cols:
        print(f"   Found DA LMP column: {da_lmp_cols[0]}")
        # Create the expected column
        df['DA_LMP_WEST_HUB'] = df[da_lmp_cols[0]]
    elif 'PJM_WESTERN_HUB_DA_LMP' in df.columns:
        print("   Creating DA_LMP_WEST_HUB from PJM_WESTERN_HUB_DA_LMP")
        df['DA_LMP_WEST_HUB'] = df['PJM_WESTERN_HUB_DA_LMP']
    else:
        print("   ERROR: No DA LMP column found!")
        print("   Available columns with 'LMP':", [col for col in df.columns if 'LMP' in col])
        raise KeyError("Cannot find DA LMP column for Western Hub")

# 3. Prepare for modeling
print("\n3. Preparing for modeling...")

# Remove rows where target is NaN
df = df[df['DA_LMP_WEST_HUB'].notna()]
print(f"   Samples with valid target: {len(df)}")

# Remove non-predictive features
features_to_remove = [
    # DA reserves
    'PJM_DA_SYNC_RESERVE',
    'PJM_DA_PRIM_RESERVE',
    
    # Real-time loads
    'PJM_RTO_LOAD',
    'PJM_AEP_LOAD', 
    'PJM_BGE_LOAD',
    'PJM_DOMINION_LOAD',
    'PJM_PECO_LOAD',
    'PJM_PENELEC_LOAD',
    'PJM_PSEG_LOAD',
    'PJM_COMED_LOAD',
    'PJM_MIDATLANTIC_LOAD',
    'MISO_LOAD',
    'NYISO_LOAD',  # Add this too
    
    # Operating reserves and capacity
    'OS_OPERATING_RESERVE_RTO',
    'OS_TOTAL_SCHED_CAP_RTO',
    'OS_UNSCHED_STM_CAP_RTO',
    'OS_UNSCHED_STM_CAP_MIDATLANTIC',
    'OS_UNSCHED_STM_CAP_DOMINION',
    'OS_UNSCHED_STM_CAP_COMED',
    'OS_UNSCHED_STM_CAP_AEP',
    'OS_UNSCHED_STM_CAP_OVEC',
    
    # Calculated reserves
    'OS_CALC_RESERVE_RTO',
    'OS_CALC_RESERVE_MIDATLANTIC',
    'OS_CALC_RESERVE_AEP',
    'OS_CALC_RESERVE_AP',
    'OS_CALC_RESERVE_ATSI',
    'OS_CALC_RESERVE_COMED',
    'OS_CALC_RESERVE_DAYTON',
    'OS_CALC_RESERVE_DEOK',
    'OS_CALC_RESERVE_DOMINION',
    'OS_CALC_RESERVE_DUQUESNE',
    'OS_CALC_RESERVE_EKPC',
    
    # Internal scheduled capacity
    'INTERNAL_SCED_CAP_RTO',
    'INTERNAL_SCED_CAP_MIDATLANTIC',
    'INTERNAL_SCED_CAP_DOMINION',
    'INTERNAL_SCED_CAP_COMED',
    'INTERNAL_SCED_CAP_AEP',
    'INTERNAL_SCED_CAP_OVEC',
    
    # Real-time generation
    'PJM_GENERATION_COAL',
    'PJM_GENERATION_GAS',
    'PJM_GENERATION_NUCLEAR',
    'PJM_GENERATION_DFO',
    'PJM_GENERATION_SOLAR_RT',
    'PJM_GENERATION_WIND',
    
    # Constraint flows
    'PJM_BEDINGTON_BLACK_OAK_FLOW',
    'PJM_500X_FLOW',
    'PJM_INTERFACE_5004_5005_FLOW',
    
    # Derived features
    'total_thermal_generation',
    
    # Current DA LMPs (what we're predicting)
    'PJM_WESTERN_HUB_DA_LMP',
    'PJM_EASTERN_HUB_DA_LMP',
    'PJM_DOMINION_HUB_DA_LMP',
    'PJM_PSEG_DA_LMP',
    'PJM_AEP_DAYTON_HUB_DA_LMP',
    'PJM_N_ILLINOIS_HUB_DA_LMP',
    'PJM_COMED_DA_LMP',
    'PJM_PECO_DA_LMP',
    'PJM_BGE_DA_LMP',
    'PJM_PENELEC_DA_LMP',
    'PJM_PEPCO_DA_LMP',
    
    # RT LMPs
    'PJM_WESTERN_HUB_RT_LMP',
    'PJM_EASTERN_HUB_RT_LMP',
    
    # Congestion (non-lagged)
    'TOT_DA_CONG_BY_ISO',
    'PJM_WESTERN_HUB_DACONG',
    'PJM_EASTERN_HUB_DACONG',
    'PJM_DOMINION_HUB_DACONG',
    'PJM_PSEG_DACONG',
    'PJM_AEP_DAYTON_HUB_DACONG',
    'PJM_N_ILLINOIS_HUB_DACONG',
    'PJM_COMED_DACONG',
    'PJM_PECO_DACONG',
    'PJM_BGE_DACONG',
    'PJM_PENELEC_DACONG',
    'PJM_PEPCO_DACONG',
]

print(f"   Removing {len(features_to_remove)} non-predictive features")
removed_count = 0
for col in features_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])
        removed_count += 1

print(f"   Actually removed: {removed_count} features")

# Separate features and target
feature_cols = [col for col in df.columns if col not in ['DA_LMP_WEST_HUB', 'is_spike']]
X = df[feature_cols]
y = df['DA_LMP_WEST_HUB']

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# Final NaN handling
if X.isna().any().any():
    print("   Handling remaining NaN values...")
    X = X.fillna(method='ffill', limit=4).fillna(method='bfill', limit=4).fillna(0)

print(f"   Final features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")


# 4. Create time splits
print("\n4. Creating time series splits...")
X = X.sort_index()
y = y.sort_index()

test_start = X.index.max() - pd.Timedelta(days=args.test_days)
val_start = test_start - pd.Timedelta(days=90)

X_test = X[X.index >= test_start].copy()
y_test = y[y.index >= test_start].copy()

X_train_full = X[X.index < test_start].copy()
y_train_full = y[y.index < test_start].copy()

X_val = X_train_full[X_train_full.index >= val_start].copy()
y_val = y_train_full[y_train_full.index >= val_start].copy()

X_train = X_train_full[X_train_full.index < val_start].copy()
y_train = y_train_full[y_train_full.index < val_start].copy()

print(f"   Train: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
print(f"   Val: {len(X_val)} samples ({X_val.index.min().date()} to {X_val.index.max().date()})")
print(f"   Test: {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")

# Calculate training years
training_years = (X_train.index.max() - X_train.index.min()).days / 365
print(f"   Training years: {training_years:.1f}")

# 5. Train advanced model
print("\n5. Training Advanced Hybrid Model...")
print("   This will train:")
print("   - Base ensemble (XGBoost, LightGBM, RandomForest)")
print("   - PatchTST (Patch Time Series Transformer)")
print("   - Extreme Event Adaptive LSTM")
print("   - Multi-scale Temporal LSTM")

model = HybridPJMPredictorAdvanced(
    spike_threshold=args.spike_threshold,
    use_deep_learning=True,
    dl_weight=args.dl_weight,
    sequence_length=args.sequence_length,
    n_dl_features=args.n_dl_features
)

try:
    model.train(
        X_train, y_train, X_val, y_val,
        use_smote=True,
        dl_epochs=args.dl_epochs,
        dl_batch_size=args.dl_batch_size
    )
    print("Training complete!")
except Exception as e:
    print(f"Training error: {str(e)}")
    import traceback
    traceback.print_exc()

# 6. Evaluate
print("\n6. Evaluating on test set...")
try:
    predictions, spike_probs = model.predict(X_test, return_spike_prob=True)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
    
    print("\n=== TEST RESULTS ===")
    print(f"MAE: ${mae:.2f}/MWh")
    print(f"RMSE: ${rmse:.2f}/MWh")
    print(f"MAPE: {mape:.1f}%")
    
    # Spike detection metrics
    actual_spikes = y_test > args.spike_threshold
    predicted_spikes = spike_probs > 0.5
    
    if actual_spikes.sum() > 0:
        tp = (actual_spikes & predicted_spikes).sum()
        spike_recall = tp / actual_spikes.sum()
        spike_precision = tp / max(predicted_spikes.sum(), 1)
        spike_f1 = 2 * (spike_precision * spike_recall) / (spike_precision + spike_recall + 1e-10)
        
        print("\n=== SPIKE DETECTION ===")
        print(f"Actual spikes: {actual_spikes.sum()}")
        print(f"Predicted spikes: {predicted_spikes.sum()}")
        print(f"Recall: {spike_recall:.1%}")
        print(f"Precision: {spike_precision:.1%}")
        print(f"F1 Score: {spike_f1:.3f}")
    
except Exception as e:
    print(f"Evaluation error: {str(e)}")
    predictions = None

# 7. Save results
print(f"\n7. Saving results to {args.output_dir}...")

# Prepare test results
test_results = {
    'mae': mae,
    'rmse': rmse,
    'mape': mape,
    'test_predictions': predictions,
    'test_actual': y_test.values,
    'timestamp': datetime.now().isoformat(),
    'training_period': {
        'train_start': str(X_train.index.min()),
        'train_end': str(X_train.index.max()),
        'val_start': str(X_val.index.min()),
        'val_end': str(X_val.index.max()),
        'test_start': str(X_test.index.min()),
        'test_end': str(X_test.index.max())
    }
}

# Use the save function from the model module
from pjm_advanced_dl_model import save_model_results
save_model_results(model, test_results, output_dir=args.output_dir)

print(f"   Model saved successfully!")

# Save predictions if available
if predictions is not None:
    results_df = pd.DataFrame({
        'datetime': X_test.index,
        'actual': y_test.values,
        'predicted': predictions,
        'error': predictions - y_test.values,
        'spike_prob': spike_probs
    })
    
    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"   Predictions saved: {csv_path}")

print("\n" + "="*70)
print("ADVANCED DL PIPELINE COMPLETE!")
print("="*70)