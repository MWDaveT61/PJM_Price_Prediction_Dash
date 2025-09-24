import sys
"""
Integrated PJM Day-Ahead LMP Prediction Model - Fully Optimized Version
========================================================================
Complete implementation optimized for AMD Ryzen 9 7950X3D
Includes all feature engineering methods with NaN handling
"""

import pandas as pd
import numpy as np
import sqlalchemy as db
import pyodbc
import urllib
from datetime import datetime, timedelta
import warnings
import pickle
import os
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, IsolationForest, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.decomposition import PCA
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# Feature Engineering
from scipy import stats
import holidays

# Set optimal thread configuration for Ryzen 9 7950X3D
N_CORES = max(1, os.cpu_count() or 1)  # Use all available logical cores
os.environ['OMP_NUM_THREADS'] = str(N_CORES)
os.environ['MKL_NUM_THREADS'] = str(N_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CORES)

# =============================================================================
# OPTIMIZED DATABASE CONNECTION AND DATA LOADING
# =============================================================================

class PJMDataPipeline:
    """
    Optimized data pipeline with parallel loading
    """
    
    def __init__(self, connection_params=None):
        if connection_params is None:
            # Default connection parameters
            self.connection_params = {
                'driver': 'ODBC Driver 18 for SQL Server',
                'server': '192.168.5.8',      # Your remote server IP
                'database': 'PowerData',
                'uid': 'davidtoole61',                    # Your SQL username
                'pwd': '2495944a'   # Your SQL password
            }
        else:
            self.connection_params = connection_params
            
        self.engine = None
        
    
    def create_connection(self):
        """Create database connection to SQL Server"""
        # Select appropriate ODBC driver
        driver = self.connection_params.get('driver')
        if sys.platform == 'darwin':
            driver = 'ODBC Driver 18 for SQL Server'
        elif not driver:
            driver = 'ODBC Driver 18 for SQL Server'
        # Build ODBC connection string
        extras = ''
        if '18' in driver:
            # ODBC 18 requires encryption by default
            extras = 'Encrypt=yes;TrustServerCertificate=yes;'
        params = urllib.parse.quote_plus(
            f'Driver={{{driver}}};'
            f'Server={self.connection_params["server"]};'
            f'Database={self.connection_params["database"]};'
            f'Uid={self.connection_params["uid"]};'
            f'Pwd={self.connection_params["pwd"]};'
            f'{extras}'
        )
        self.engine = db.create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        return self.engine

    
    def load_comprehensive_data(self, start_date='2018-01-01', end_date='2025-09-24'):
        """
        Load all required PJM data from database
        
        Returns:
        --------
        pd.DataFrame : Master dataframe with all features
        """
        
        if self.engine is None:
            self.create_connection()
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Loading PJM data from {start_date} to {end_date}...")
        
        # Create master dataframe
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        master_df = pd.DataFrame(index=date_range)
        master_df.index.name = 'datetime'
        
        # Define all data queries
        data_queries = self._get_data_queries()
        
        # Load each data type
        for category, queries in data_queries.items():
            print(f"\nLoading {category} data...")
            for col_name, (data_type, location) in queries.items():
                try:
                    df = self._load_single_data_type(data_type, location, start_date, end_date)
                    if not df.empty:
                        master_df[col_name] = df['Value'].reindex(master_df.index)
                        non_null = master_df[col_name].notna().sum()
                        if non_null > 0:
                            print(f"  {col_name}: {non_null} records")
                except Exception as e:
                    print(f"  Warning: Could not load {col_name}: {str(e)}")
        
        self.engine.dispose()
        print("\nColumns loaded by category:")
        congestion_cols = [col for col in master_df.columns if 'CONG' in col.upper()]
        print(f"  Congestion columns: {congestion_cols}")
        if 'TOT_DA_CONG_BY_ISO' in master_df.columns:
            non_null = master_df['TOT_DA_CONG_BY_ISO'].notna().sum()
            print(f"  TOT_DA_CONG_BY_ISO has {non_null} non-null values")
        else:
            print("  WARNING: TOT_DA_CONG_BY_ISO not found in loaded data!")
        # Remove rows that are all NaN
        master_df = master_df.dropna(how='all')
        
        print(f"\nData loading complete: {len(master_df.columns)} columns, {len(master_df)} rows")
        
        return master_df
    
    def _load_single_data_type(self, data_type, location, start_date, end_date):
        """Load a single data type from database"""
        
        query = f"""
        SELECT 
            CAST(ed.DateTime AS DATETIME) as DateTime,
            ed.Value
        FROM EnergyData ed
            INNER JOIN DataTypeLocations dtl ON ed.DataTypeLocationID = dtl.DataTypeLocationID
            INNER JOIN DataTypes dt ON dtl.DataTypeID = dt.DataTypeID
            INNER JOIN Locations l ON dtl.LocationID = l.LocationID
        WHERE 
            dt.DataTypeCode = '{data_type}'
            AND l.LocationCode = '{location}'
            AND ed.DateTime >= '{start_date}'
            AND ed.DateTime <= '{end_date}'
        ORDER BY ed.DateTime
        """
        
        df = pd.read_sql(query, self.engine, parse_dates=['DateTime'], index_col='DateTime')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        return df
    
    def _get_data_queries(self):
        """Define all data queries organized by category"""
        all_weather_locations = [
            'Washington DC', 'Chicago Midway', 'Baltimore', 'Dayton', 
            'Philly', 'Richmond', 'Atlanta'
        ]
        
        return {
            'weather': {
                # Temperature features
                **{f'TEMP_F_{loc.replace(" ", "_")}': ('TEMP_F', loc) 
                   for loc in all_weather_locations},
                # Cloud cover
                **{f'CLOUD_COVER_{loc.replace(" ", "_")}': ('CLOUD_COVER_PCT', loc) 
                   for loc in all_weather_locations},
                # Dew point
                **{f'DEWPOINT_{loc.replace(" ", "_")}': ('NWSDEWP_C', loc) 
                   for loc in all_weather_locations},
                # Precipitation
                **{f'PRECIP_{loc.replace(" ", "_")}': ('PRECIP_LWC', loc) 
                   for loc in all_weather_locations},
            },
            'gas_prices': {
                'GAS_Henry_Hub': ('GAS_PRICE_AVG', 'Henry Hub'),
                'GAS_Dom_South': ('GAS_PRICE_AVG', 'Dom South'),
                'GAS_M3': ('GAS_PRICE_AVG', 'M3'),
                'GAS_TCO': ('GAS_PRICE_AVG', 'TCO'),
                'GAS_Chicago_Citygate': ('GAS_PRICE_AVG', 'Chicago Citygate'),
                'GAS_Transco_Z6': ('GAS_PRICE_AVG', 'Transco-Z6 (non-NY)'),
                'GAS_Transco_Z5': ('GAS_PRICE_AVG', 'Transco Zn5'),
            },
            'prices': {
                'PJM_WESTERN_HUB_DA_LMP': ('DALMP', 'WESTERN HUB'),
                'PJM_EASTERN_HUB_DA_LMP': ('DALMP', 'EASTERN HUB'),
                'PJM_DOMINION_HUB_DA_LMP': ('DALMP', 'DOMINION HUB'),
                'PJM_PSEG_DA_LMP': ('DALMP', 'PSEG'),
                'PJM_AEP_DAYTON_HUB_DA_LMP': ('DALMP', 'AEP-DAYTON HUB'),
                'PJM_N_ILLINOIS_HUB_DA_LMP': ('DALMP', 'N ILLINOIS HUB'),
                'PJM_COMED_DA_LMP': ('DALMP', 'COMED'),
                'PJM_PECO_DA_LMP': ('DALMP', 'PECO'),
                'PJM_BGE_DA_LMP': ('DALMP', 'BGE'),
                'PJM_WESTERN_HUB_RT_LMP': ('RTLMP', 'WESTERN HUB'),
                'PJM_EASTERN_HUB_RT_LMP': ('RTLMP', 'EASTERN HUB'),
            },
            'load': {
                # PJM loads
                'PJM_RTO_LOAD': ('RTLOAD', 'RTO COMBINED'),
                'PJM_RTO_LOAD_FORECAST_DA': ('BIDCLOSE_LOAD_FORECAST', 'RTO COMBINED'),
                'PJM_MIDATLANTIC_LOAD_FORECAST_DA': ('BIDCLOSE_LOAD_FORECAST', 'MIDATLANTIC'),
                'PJM_COMED_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'COMED'),
                'PJM_AEP_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'AEP'),
                'PJM_PSEG_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'PSEG'),
                'PJM_DOMINION_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'DOMINION'),
                'PJM_PECO_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'PECO'),
                'PJM_BGE_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'BGE'),
                'PJM_PENELEC_LOAD_FORECAST': ('BIDCLOSE_LOAD_FORECAST', 'PENELEC'),
                
                # Zonal loads
                'PJM_COMED_LOAD': ('RTLOAD', 'COMED'),
                'PJM_AEP_LOAD': ('RTLOAD', 'AEP'),
                'PJM_PSEG_LOAD': ('RTLOAD', 'PSEG'),
                'PJM_DOMINION_LOAD': ('RTLOAD', 'DOMINION'),
                'PJM_PECO_LOAD': ('RTLOAD', 'PECO'),
                'PJM_BGE_LOAD': ('RTLOAD', 'BGE'),
                'PJM_PENELEC_LOAD': ('RTLOAD', 'PENELEC'),
                'PJM_MIDATLANTIC_LOAD': ('RTLOAD', 'MIDATLANTIC'),
                
                # Neighboring ISOs
                'NYISO_LOAD_FORECAST_DA': ('LOAD_FORECAST', 'NYISO'),
                'MISO_LOAD_FORECAST_DA': ('BIDCLOSE_LOAD_FORECAST', 'MISO'),
                'NYISO_LOAD': ('RTLOAD', 'NYISO'),
                'MISO_LOAD': ('RTLOAD', 'MISO'),
            },
            'generation': {
                'PJM_GENERATION_COAL': ('COALGEN_HOURLY', 'PJMISO'),
                'PJM_GENERATION_GAS': ('GASGEN_HOURLY', 'PJMISO'),
                'PJM_GENERATION_NUCLEAR': ('NUCLEARGEN_HOURLY', 'PJMISO'),
                'PJM_GENERATION_DFO': ('DFOGEN_HOURLY', 'PJMISO'),
                'PJM_GENERATION_SOLAR_RT': ('GENERATION_SOLAR_RT', 'RTO COMBINED'),
                'PJM_GENERATION_WIND': ('WIND_RTF', 'RTO COMBINED'),
                'PJM_SOLARFCST_HOURLY': ('SOLARFCST_HOURLY', 'RTO COMBINED'),
                'PJM_HRLY_SOLAR_FCST_BTM': ('HRLY_SOLAR_FCST_BTM', 'RTO COMBINED'),
            },
            'reserves': {
                'PJM_DA_SYNC_RESERVE': ('DA_MCP_SYNC_RESERVE', 'RTO COMBINED'),
                'PJM_DA_PRIM_RESERVE': ('DA_MCP_PRIM_RESERVE', 'RTO COMBINED'),
            },
            'constraints': {
                'PJM_BEDINGTON_BLACK_OAK_FLOW': ('PJM_FLOW_PERC_OF_LIMIT', 'BEDINGTON - BLACK OAK'),
                'PJM_500X_FLOW': ('PJM_FLOW_PERC_OF_LIMIT', '500X'),
                'PJM_INTERFACE_5004_5005_FLOW': ('PJM_FLOW_PERC_OF_LIMIT', '5004/5005 INTERFACE'),
            },
            'operating_reserves': {
                'OS_OPERATING_RESERVE_RTO': ('OS_OPERATING_RESERVE', 'RTO COMBINED'),
                'OS_CALC_RESERVE_RTO': ('OS_CALC_OPERATING_RESERVE', 'RTO COMBINED'),
                'OS_CALC_RESERVE_MIDATLANTIC': ('OS_CALC_OPERATING_RESERVE', 'MID-ATLANTIC REGION'),
                'OS_CALC_RESERVE_AEP': ('OS_CALC_OPERATING_RESERVE', 'AEP'),
                'OS_CALC_RESERVE_AP': ('OS_CALC_OPERATING_RESERVE', 'AP'),
                'OS_CALC_RESERVE_ATSI': ('OS_CALC_OPERATING_RESERVE', 'ATSI'),
                'OS_CALC_RESERVE_COMED': ('OS_CALC_OPERATING_RESERVE', 'COMED'),
                'OS_CALC_RESERVE_DAYTON': ('OS_CALC_OPERATING_RESERVE', 'DAYTON'),
                'OS_CALC_RESERVE_DEOK': ('OS_CALC_OPERATING_RESERVE', 'DEOK'),
                'OS_CALC_RESERVE_DOMINION': ('OS_CALC_OPERATING_RESERVE', 'DOMINION'),
                'OS_CALC_RESERVE_DUQUESNE': ('OS_CALC_OPERATING_RESERVE', 'DUQUESNE'),
                'OS_CALC_RESERVE_EKPC': ('OS_CALC_OPERATING_RESERVE', 'EKPC')
            },
            'capacity_data': {
                'OS_TOTAL_SCHED_CAP_RTO': ('OS_TOTAL_SCHED_CAP', 'RTO COMBINED'),
                'OS_UNSCHED_STM_CAP_RTO': ('OS_UNSCHED_STM_CAP', 'RTO COMBINED'),
                'OS_UNSCHED_STM_CAP_MIDATLANTIC': ('OS_UNSCHED_STM_CAP', 'MID-ATLANTIC REGION'),
                'OS_UNSCHED_STM_CAP_DOMINION': ('OS_UNSCHED_STM_CAP', 'DOMINION'),
                'OS_UNSCHED_STM_CAP_COMED': ('OS_UNSCHED_STM_CAP', 'COMED'),
                'OS_UNSCHED_STM_CAP_AEP': ('OS_UNSCHED_STM_CAP', 'AEP'),
                'OS_UNSCHED_STM_CAP_OVEC': ('OS_UNSCHED_STM_CAP', 'OVEC'),
                'INTERNAL_SCED_CAP_RTO': ('INTERNAL_SCED_CAP', 'RTO COMBINED'),
                'INTERNAL_SCED_CAP_MIDATLANTIC': ('INTERNAL_SCED_CAP', 'MID-ATLANTIC REGION'),
                'INTERNAL_SCED_CAP_DOMINION': ('INTERNAL_SCED_CAP', 'DOMINION'),
                'INTERNAL_SCED_CAP_COMED': ('INTERNAL_SCED_CAP', 'COMED'),
                'INTERNAL_SCED_CAP_AEP': ('INTERNAL_SCED_CAP', 'AEP'),
                'INTERNAL_SCED_CAP_OVEC': ('INTERNAL_SCED_CAP', 'OVEC'),
            },
            'congestion': {
                'TOT_DA_CONG_BY_ISO': ('TOT_DA_CONG_BY_ISO', 'PJMISO'),
                'PJM_WESTERN_HUB_DACONG': ('DACONG', 'WESTERN HUB'),
                'PJM_EASTERN_HUB_DACONG': ('DACONG', 'EASTERN HUB'),
                'PJM_DOMINION_HUB_DACONG': ('DACONG', 'DOMINION HUB'),
                'PJM_PSEG_DACONG': ('DACONG', 'PSEG'),
                'PJM_AEP_DAYTON_HUB_DACONG': ('DACONG', 'AEP-DAYTON HUB'),
                'PJM_N_ILLINOIS_HUB_DACONG': ('DACONG', 'N ILLINOIS HUB'),
                'PJM_COMED_DACONG': ('DACONG', 'COMED'),
                'PJM_PECO_DACONG': ('DACONG', 'PECO'),
                'PJM_BGE_DACONG': ('DACONG', 'BGE'),
            }
        }
# =============================================================================
# ENHANCED FEATURE ENGINEERING WITH COMPLETE NAN HANDLING
# =============================================================================

class EnhancedFeatureEngineer:
    """
    Optimized feature engineering with parallel processing and complete NaN handling
    """
    
    def __init__(self):
        self.us_holidays = holidays.US()
        self.scaler = RobustScaler()
        self.spike_threshold = 175
        
    def create_all_features(self, df):
        """
        Create comprehensive feature set with complete NaN handling
        """
        
        print("Creating comprehensive feature set...")
        start_time = time.time()
        
        # Ensure target column exists
        if 'PJM_WESTERN_HUB_DA_LMP' in df.columns:
            df['DA_LMP_WEST_HUB'] = df['PJM_WESTERN_HUB_DA_LMP']
        
        # Process features in order of dependencies
        df = self._add_temporal_features(df)
        df = self._add_weather_features(df)
        df = self._add_price_features(df)
        df = self._add_solar_ramping_features(df)
        df = self._add_enhanced_load_features(df)
        df = self._add_net_load_features(df)
        df = self._add_price_divergence_features(df)
        df = self._add_gas_price_features(df)
        
        # Add capacity and reserve features
        df = self._add_capacity_adequacy_features(df)
        df = self._add_operating_reserve_features(df)
        
        # System stress features
        df = self._add_system_stress_features(df)
        df = self._add_enhanced_system_stress_features(df)
        df = self._add_load_shape_features(df)
        df = self._add_regional_features(df)
        df = self._add_cross_feature_interactions(df)
        df = self._add_advanced_time_features(df)
        
        # Lag and rolling features
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_spike_features(df)
        
        # ADD CONGESTION FEATURES HERE
        df = self._add_congestion_features(df)
        
        # CRITICAL: Complete NaN handling before returning
        df = self._complete_nan_handling(df)
        
        elapsed = time.time() - start_time
        print(f"Feature engineering complete: {len(df.columns)} features in {elapsed:.2f} seconds")
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
        
        return df
    
    
    def _add_congestion_features(self, df):
        """Add congestion features using DACONG (congestion component of DA LMP)"""
        
        # Get DACONG columns
        dacong_cols = [col for col in df.columns if 'DACONG' in col]
        
        if dacong_cols:
            print("  Adding congestion features from DACONG data...")
            
            # For each hub's congestion
            for col in dacong_cols:
                # Create lagged versions (CRITICAL for DA prediction)
                df[f'{col}_lag24'] = df[col].shift(24)
                df[f'{col}_lag48'] = df[col].shift(48)
                df[f'{col}_lag168'] = df[col].shift(168)
                
                # Week-over-week change
                df[f'{col}_wow_change'] = df[f'{col}_lag24'] - df[f'{col}_lag168']
                
                # Rolling statistics
                df[f'{col}_ma7d'] = df[col].shift(24).rolling(168, min_periods=24).mean()
                df[f'{col}_std7d'] = df[col].shift(24).rolling(168, min_periods=24).std()
            
            # System-wide congestion metrics
            if 'PJM_WESTERN_HUB_DACONG' in df.columns:
                # Western hub congestion patterns
                df['west_congestion_lag24'] = df['PJM_WESTERN_HUB_DACONG'].shift(24)
                df['west_congestion_lag168'] = df['PJM_WESTERN_HUB_DACONG'].shift(168)
                
                # High congestion indicator
                if len(df) > 720:
                    cong_90th = df['PJM_WESTERN_HUB_DACONG'].shift(24).rolling(720, min_periods=168).quantile(0.9)
                    df['high_congestion_west'] = (df['west_congestion_lag24'] > cong_90th).astype(int)
            
            # Inter-hub congestion spreads
            if 'PJM_WESTERN_HUB_DACONG' in df.columns and 'PJM_EASTERN_HUB_DACONG' in df.columns:
                df['west_east_congestion_spread_lag24'] = (
                    df['PJM_WESTERN_HUB_DACONG'].shift(24) - 
                    df['PJM_EASTERN_HUB_DACONG'].shift(24)
                )
            
            # Average system congestion
            if len(dacong_cols) > 1:
                # Create lagged versions first
                dacong_lag_cols = [f'{col}_lag24' for col in dacong_cols if f'{col}_lag24' in df.columns]
                if dacong_lag_cols:
                    df['system_avg_congestion_lag24'] = df[dacong_lag_cols].mean(axis=1)
                    df['congestion_dispersion_lag24'] = df[dacong_lag_cols].std(axis=1)
                    
                    # Congestion concentration
                    df['congestion_concentration'] = self._safe_divide(
                        df['congestion_dispersion_lag24'],
                        df['system_avg_congestion_lag24'].abs() + 1,
                        fill_value=0
                    )
            
            print(f"    Added {sum([1 for col in df.columns if 'cong' in col.lower()])} congestion features")
        
        else:
            print("  Warning: No DACONG data found")
        
        return df
    
    def _complete_nan_handling(self, df):
        """
        Comprehensive NaN handling to ensure no NaN values remain
        """
        
        print(f"  Starting NaN handling with {len(df.columns)} features...")
        initial_cols = len(df.columns)
        
        # First pass: replace infinities
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Check how many columns have too many NaNs
        nan_percentages = df.isna().sum() / len(df)
        high_nan_cols = nan_percentages[nan_percentages > 0.5]
        if len(high_nan_cols) > 0:
            print(f"  Warning: {len(high_nan_cols)} columns have >50% NaN values")
            print(f"    Sample high-NaN columns: {high_nan_cols.head().index.tolist()}")
        
        # Second pass: forward fill then backward fill with limits
        df = df.fillna(method='ffill', limit=4)
        df = df.fillna(method='bfill', limit=4)
        # Third pass: fill remaining NaNs with column-specific strategies
        for col in df.columns:
            if df[col].isna().any():
                # Get column statistics before filling
                col_mean = df[col].mean() if df[col].notna().any() else 0
                col_median = df[col].median() if df[col].notna().any() else 0
                
                # Fill based on column type
                if 'lag' in col.lower() or 'ma' in col.lower():
                    df[col] = df[col].fillna(col_mean)
                elif 'ratio' in col.lower() or 'pct' in col.lower():
                    df[col] = df[col].fillna(0)
                elif 'norm' in col.lower():
                    df[col] = df[col].fillna(1)
                elif any(x in col.lower() for x in ['price', 'lmp', 'cost']):
                    df[col] = df[col].fillna(col_median)
                else:
                    df[col] = df[col].fillna(0)
        
        # Fourth pass: cap extreme values
        for col in numeric_cols:
            # Calculate reasonable bounds
            if df[col].notna().any():
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                iqr = q99 - q01
                lower_bound = q01 - 3 * iqr
                upper_bound = q99 + 3 * iqr
                
                # Apply bounds but not too restrictive
                lower_bound = max(lower_bound, -1e6)
                upper_bound = min(upper_bound, 1e6)
                
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Final verification
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"  Warning: Final NaN fill for {len(nan_cols)} columns")
            for col in nan_cols:
                df[col] = df[col].fillna(0)
        
        # Verify no NaN or inf values remain
        assert not df.isna().any().any(), "NaN values still present after cleaning"
        assert not np.isinf(df.select_dtypes(include=[np.number])).any().any(), "Inf values still present"
        
        print(f"  NaN handling complete - {len(df.columns)} features retained (lost {initial_cols - len(df.columns)})")
    
        return df
    
    def _safe_divide(self, numerator, denominator, fill_value=0):
        """Safely divide arrays handling zeros and infinities"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / (denominator + 1e-10)
            result = np.where(np.isfinite(result), result, fill_value)
            result = np.clip(result, -1e6, 1e6)
        return result
    
    def _add_temporal_features(self, df):
        """Enhanced temporal features with cyclical encoding"""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Basic temporal
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # Business features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_holiday'] = df.index.map(lambda x: x in self.us_holidays).astype(int)
        df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] < 23) & 
                         (df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
        
        # Season
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] >= 11)).astype(int)
        
        return df
    
    def _add_weather_features(self, df):
        """Add weather-based features for electricity demand prediction"""
        
        # Get temperature columns
        temp_cols = [col for col in df.columns if 'TEMP_F_' in col]
        
        if temp_cols:
            # Average temperature across region
            df['temp_avg'] = df[temp_cols].mean(axis=1)
            df['temp_max'] = df[temp_cols].max(axis=1)
            df['temp_min'] = df[temp_cols].min(axis=1)
            df['temp_spread'] = df['temp_max'] - df['temp_min']
            
            # Heating/Cooling degree days
            df['heating_degrees'] = np.maximum(65 - df['temp_avg'], 0)
            df['cooling_degrees'] = np.maximum(df['temp_avg'] - 65, 0)
            
            # Extreme temperature flags
            df['extreme_cold'] = (df['temp_avg'] < 20).astype(int)
            df['extreme_heat'] = (df['temp_avg'] > 95).astype(int)
            
            # Temperature change rates
            df['temp_change_1h'] = df['temp_avg'].diff(1)
            df['temp_change_24h'] = df['temp_avg'].diff(24)
            
            # Lagged weather (for prediction)
            df['temp_avg_lag24'] = df['temp_avg'].shift(24)
            df['cooling_degrees_lag24'] = df['cooling_degrees'].shift(24)
            df['heating_degrees_lag24'] = df['heating_degrees'].shift(24)
        
        # Cloud cover impacts solar generation
        cloud_cols = [col for col in df.columns if 'CLOUD_COVER_' in col]
        if cloud_cols:
            df['cloud_cover_avg'] = df[cloud_cols].mean(axis=1)
            df['cloud_cover_lag24'] = df['cloud_cover_avg'].shift(24)
            
            # High cloud cover reduces solar
            df['cloudy_day'] = (df['cloud_cover_avg'] > 70).astype(int)
        
        # Humidity from dewpoint
        dewpoint_cols = [col for col in df.columns if 'DEWPOINT_' in col]
        if dewpoint_cols and temp_cols:
            # Convert dewpoint C to F
            df['dewpoint_avg_c'] = df[dewpoint_cols].mean(axis=1)
            df['dewpoint_avg_f'] = df['dewpoint_avg_c'] * 9/5 + 32
            
            # Relative humidity approximation
            df['humidity_index'] = self._safe_divide(df['dewpoint_avg_f'], df['temp_avg'], fill_value=0.5)
            df['high_humidity'] = (df['humidity_index'] > 0.7).astype(int)
        
        # Precipitation
        precip_cols = [col for col in df.columns if 'PRECIP_' in col]
        if precip_cols:
            df['precip_total'] = df[precip_cols].sum(axis=1)
            df['is_raining'] = (df['precip_total'] > 0.1).astype(int)
            df['heavy_rain'] = (df['precip_total'] > 1.0).astype(int)
        
        # Combined weather stress
        weather_stress = []
        if 'extreme_heat' in df.columns:
            weather_stress.append(df['extreme_heat'])
        if 'extreme_cold' in df.columns:
            weather_stress.append(df['extreme_cold'])
        if 'high_humidity' in df.columns and 'extreme_heat' in df.columns:
            weather_stress.append(df['high_humidity'] * df['extreme_heat'])  # Heat index
        
        if weather_stress:
            df['weather_stress_score'] = sum(weather_stress) / len(weather_stress)
        
        return df
    
    def _add_price_features(self, df):
        """Price spreads and congestion indicators"""
        # Create lagged versions of DA prices first
        da_lmp_cols = [col for col in df.columns if 'DA_LMP' in col and 'lag' not in col]
        
        # Create 24-hour lagged versions of all DA prices
        for col in da_lmp_cols:
            df[f'{col}_LAG24'] = df[col].shift(24)
            df[f'{col}_LAG48'] = df[col].shift(48)
            df[f'{col}_LAG168'] = df[col].shift(168)
        
        # Hub-to-hub spreads - using LAGGED prices
        if 'PJM_WESTERN_HUB_DA_LMP_LAG24' in df.columns:
            if 'PJM_EASTERN_HUB_DA_LMP_LAG24' in df.columns:
                df['west_east_spread_lag24'] = df['PJM_WESTERN_HUB_DA_LMP_LAG24'] - df['PJM_EASTERN_HUB_DA_LMP_LAG24']
            if 'PJM_DOMINION_HUB_DA_LMP_LAG24' in df.columns:
                df['west_dom_spread_lag24'] = df['PJM_WESTERN_HUB_DA_LMP_LAG24'] - df['PJM_DOMINION_HUB_DA_LMP_LAG24']
            if 'PJM_N_ILLINOIS_HUB_DA_LMP_LAG24' in df.columns:
                df['west_nillinois_spread_lag24'] = df['PJM_WESTERN_HUB_DA_LMP_LAG24'] - df['PJM_N_ILLINOIS_HUB_DA_LMP_LAG24']
        
        return df
    
    def _add_solar_ramping_features(self, df):
        """Critical solar ramping features"""
        solar_col = None
        for col in ['PJM_GENERATION_SOLAR_RT', 'PJM_GENERATION_SOLAR', 'GENERATION_SOLAR_RT']:
            if col in df.columns:
                solar_col = col
                break
        
        if solar_col:
            df['solar_gen'] = df[solar_col].fillna(0)
            df['solar_ramp_1h'] = df['solar_gen'].diff(1)
            df['solar_ramp_2h'] = df['solar_gen'].diff(2)
            df['solar_ramp_3h'] = df['solar_gen'].diff(3)
            
            # Solar penetration
            if 'PJM_RTO_LOAD' in df.columns:
                df['solar_penetration'] = self._safe_divide(df['solar_gen'], df['PJM_RTO_LOAD'], fill_value=0)
        
        return df
    
    def _add_net_load_features(self, df):
        """Net load features"""
        if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            load_col = 'PJM_RTO_LOAD_FORECAST_DA'
        elif 'PJM_RTO_LOAD' in df.columns:
            load_col = 'PJM_RTO_LOAD'
        else:
            return df
        
        renewable_cols = []
        if 'solar_gen' in df.columns:
            renewable_cols.append('solar_gen')
        if 'PJM_GENERATION_WIND' in df.columns:
            renewable_cols.append('PJM_GENERATION_WIND')
        
        if renewable_cols:
            df['total_renewable'] = df[renewable_cols].sum(axis=1)
        else:
            df['total_renewable'] = 0
        
        df['net_load'] = df[load_col] - df['total_renewable']
        df['net_load_ramp_1h'] = df['net_load'].diff(1)
        df['net_load_ramp_3h'] = df['net_load'].diff(3)
        
        return df
    
    def _add_price_divergence_features(self, df):
        """Price divergence analysis"""
        
        lagged_da_cols = [col for col in df.columns if 'DA_LMP' in col and 'LAG24' in col]
        
        if len(lagged_da_cols) < 3:
            return df
        
        # Regional averages using the lagged columns
        eastern_lag_nodes = [col for col in lagged_da_cols if any(x in col for x in ['PSEG', 'PECO', 'EASTERN_HUB', 'BGE'])]
        western_lag_nodes = [col for col in lagged_da_cols if any(x in col for x in ['COMED', 'N_ILLINOIS', 'WESTERN_HUB', 'AEP'])]
        
        if eastern_lag_nodes:
            df['eastern_avg_price_lag24'] = df[eastern_lag_nodes].mean(axis=1)
        if western_lag_nodes:
            df['western_avg_price_lag24'] = df[western_lag_nodes].mean(axis=1)
        if eastern_lag_nodes and western_lag_nodes:
            df['east_west_avg_spread_lag24'] = df['eastern_avg_price_lag24'] - df['western_avg_price_lag24']
        
        # Actually use lagged_da_cols for overall statistics
        if lagged_da_cols:
            # System-wide price statistics from lagged DA prices
            df['system_avg_price_lag24'] = df[lagged_da_cols].mean(axis=1)
            df['price_dispersion_lag24'] = df[lagged_da_cols].std(axis=1)
            df['max_price_spread_lag24'] = df[lagged_da_cols].max(axis=1) - df[lagged_da_cols].min(axis=1)
            
            # Coefficient of variation (normalized dispersion)
            df['price_cv_lag24'] = self._safe_divide(df['price_dispersion_lag24'], df['system_avg_price_lag24'], fill_value=0)
        
        return df
    
    def _add_gas_price_features(self, df):
        """Gas price features"""
        gas_hubs = ['GAS_Henry_Hub', 'GAS_Dom_South', 'GAS_M3', 'GAS_TCO', 'GAS_Chicago_Citygate']
        
        for hub in gas_hubs:
            if hub in df.columns:
                df[f'{hub}_lag1d'] = df[hub].shift(24)
                df[f'{hub}_lag7d'] = df[hub].shift(168)
        
        return df
    
    def _add_lag_features(self, df):
        """Lagged features"""
        if 'DA_LMP_WEST_HUB' in df.columns:
            for lag in [24, 48, 72, 168, 336, 720]:
                df[f'price_lag_{lag}h'] = df['DA_LMP_WEST_HUB'].shift(lag)
        
        key_features = ['net_load', 'solar_gen', 'system_stress_score', 'temp_avg', 'load_forecast_error']
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_lag24'] = df[feat].shift(24)
                df[f'{feat}_lag168'] = df[feat].shift(168)
        
        return df
    
    def _add_rolling_features(self, df):
        """Rolling window statistics - vectorized for speed"""
        windows = [24, 48, 72, 168, 336]
        
        if 'DA_LMP_WEST_HUB' in df.columns:
            # Vectorized rolling computations
            price_shifted = df['DA_LMP_WEST_HUB'].shift(24)
            for window in windows:
                df[f'price_ma{window}'] = price_shifted.rolling(window, min_periods=1).mean()
                df[f'price_std{window}'] = price_shifted.rolling(window, min_periods=1).std()
        
        return df
    
    def _add_spike_features(self, df):
        """Spike detection features"""
        if 'DA_LMP_WEST_HUB' not in df.columns:
            return df
        
        df['is_spike'] = (df['DA_LMP_WEST_HUB'] > self.spike_threshold).astype(int)
        df['hours_since_spike'] = (~df['is_spike']).astype(int).groupby(df['is_spike'].cumsum()).cumsum()
        
        return df

    # Continue with enhanced load features...
    def _add_enhanced_load_features(self, df):
        """Add comprehensive load features using FORECASTS for prediction"""
        
        # 1. PJM Load forecast accuracy (using RT load for historical error analysis)
        if 'PJM_RTO_LOAD' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['load_forecast_error'] = df['PJM_RTO_LOAD'] - df['PJM_RTO_LOAD_FORECAST_DA']
            df['load_forecast_error_pct'] = self._safe_divide(df['load_forecast_error'], df['PJM_RTO_LOAD_FORECAST_DA'], 0) * 100
            df['load_forecast_error_abs'] = np.abs(df['load_forecast_error'])
            
            # Lagged errors (systematic bias)
            df['load_forecast_error_lag24'] = df['load_forecast_error'].shift(24)
            df['load_forecast_error_lag168'] = df['load_forecast_error'].shift(168)
            
            # Rolling bias detection
            df['load_forecast_bias_7d'] = df['load_forecast_error'].shift(24).rolling(168, min_periods=1).mean()
        
        # 2. Zonal load patterns using FORECASTS
        zonal_forecasts = {
            'comed': 'PJM_COMED_LOAD_FORECAST',
            'aep': 'PJM_AEP_LOAD_FORECAST',
            'pseg': 'PJM_PSEG_LOAD_FORECAST',
            'dominion': 'PJM_DOMINION_LOAD_FORECAST',
            'peco': 'PJM_PECO_LOAD_FORECAST',
            'bge': 'PJM_BGE_LOAD_FORECAST',
            'penelec': 'PJM_PENELEC_LOAD_FORECAST'
        }
        
        available_zone_forecasts = [v for k, v in zonal_forecasts.items() if v in df.columns]
        
        if available_zone_forecasts and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            # Zonal concentration using FORECASTS
            for zone_name, forecast_col in zonal_forecasts.items():
                if forecast_col in df.columns:
                    df[f'{zone_name}_load_share_forecast'] = self._safe_divide(df[forecast_col], df['PJM_RTO_LOAD_FORECAST_DA'], 0)
                    
                    # Calculate zonal forecast error if RT load available
                    rt_col = f'PJM_{zone_name.upper()}_LOAD'
                    if rt_col in df.columns:
                        df[f'{zone_name}_forecast_error'] = df[rt_col] - df[forecast_col]
                        df[f'{zone_name}_forecast_error_lag24'] = df[f'{zone_name}_forecast_error'].shift(24)
            
            # Load imbalance using forecasts
            if len(available_zone_forecasts) > 1:
                zone_mean = df[available_zone_forecasts].mean(axis=1)
                zone_std = df[available_zone_forecasts].std(axis=1)
                df['load_imbalance_forecast'] = self._safe_divide(zone_std, zone_mean, 0)
                df['max_zonal_load_forecast'] = df[available_zone_forecasts].max(axis=1)
                df['min_zonal_load_forecast'] = df[available_zone_forecasts].min(axis=1)
                
                # East-West imbalance
                eastern_zones = ['pseg', 'peco', 'bge', 'penelec']
                western_zones = ['comed', 'aep']
                
                eastern_forecast_cols = [zonal_forecasts[z] for z in eastern_zones if z in zonal_forecasts and zonal_forecasts[z] in df.columns]
                western_forecast_cols = [zonal_forecasts[z] for z in western_zones if z in zonal_forecasts and zonal_forecasts[z] in df.columns]
                
                if eastern_forecast_cols:
                    df['eastern_load_forecast'] = df[eastern_forecast_cols].sum(axis=1)
                if western_forecast_cols:
                    df['western_load_forecast'] = df[western_forecast_cols].sum(axis=1)
                if eastern_forecast_cols and western_forecast_cols:
                    df['east_west_load_ratio_forecast'] = self._safe_divide(df['eastern_load_forecast'], df['western_load_forecast'], 1)
        
        # 3. Neighboring ISO impacts
        if 'NYISO_LOAD_FORECAST_DA' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['nyiso_pjm_load_ratio'] = self._safe_divide(df['NYISO_LOAD_FORECAST_DA'], df['PJM_RTO_LOAD_FORECAST_DA'], 0)
            df['nyiso_load_lag24'] = df['NYISO_LOAD_FORECAST_DA'].shift(24)
            df['nyiso_load_change_24h'] = df['NYISO_LOAD_FORECAST_DA'].diff(24)
        
        if 'MISO_LOAD_FORECAST_DA' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['miso_pjm_load_ratio'] = self._safe_divide(df['MISO_LOAD_FORECAST_DA'], df['PJM_RTO_LOAD_FORECAST_DA'], 0)
            df['miso_load_lag24'] = df['MISO_LOAD_FORECAST_DA'].shift(24)
            df['miso_load_change_24h'] = df['MISO_LOAD_FORECAST_DA'].diff(24)
        
        # 4. Load ramping features
        if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['load_ramp_1h'] = df['PJM_RTO_LOAD_FORECAST_DA'].diff(1)
            df['load_ramp_2h'] = df['PJM_RTO_LOAD_FORECAST_DA'].diff(2)
            df['load_ramp_4h'] = df['PJM_RTO_LOAD_FORECAST_DA'].diff(4)
            
            # Ramp rates
            df['load_ramp_rate_1h'] = self._safe_divide(df['load_ramp_1h'], df['PJM_RTO_LOAD_FORECAST_DA'].shift(1), 0) * 100
            
            # Morning/evening ramps
            df['morning_ramp'] = np.where(
                df['hour'].isin([6, 7, 8, 9]),
                df['load_ramp_1h'],
                0
            )
            df['evening_ramp'] = np.where(
                df['hour'].isin([17, 18, 19, 20]),
                df['load_ramp_1h'],
                0
            )
        
        # 5. Load percentiles and extremes
        if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['load_percentile_24h'] = df['PJM_RTO_LOAD_FORECAST_DA'].rolling(24, min_periods=1).rank(pct=True)
            df['load_percentile_168h'] = df['PJM_RTO_LOAD_FORECAST_DA'].rolling(168, min_periods=1).rank(pct=True)
            
            # High/low load indicators
            load_90th = df['PJM_RTO_LOAD_FORECAST_DA'].rolling(720, min_periods=24).quantile(0.9)
            load_10th = df['PJM_RTO_LOAD_FORECAST_DA'].rolling(720, min_periods=24).quantile(0.1)
            df['high_load'] = (df['PJM_RTO_LOAD_FORECAST_DA'] > load_90th).astype(int)
            df['low_load'] = (df['PJM_RTO_LOAD_FORECAST_DA'] < load_10th).astype(int)
        
        # 6. Reserve margin proxy
        if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            historical_peak = df['PJM_RTO_LOAD_FORECAST_DA'].rolling(8760, min_periods=720).max()
            df['load_vs_historical_peak'] = self._safe_divide(df['PJM_RTO_LOAD_FORECAST_DA'], historical_peak, 0.5)
            df['reserve_margin_proxy'] = 1 - df['load_vs_historical_peak']
            df['capacity_stress'] = (df['load_vs_historical_peak'] > 0.85).astype(int)
        
        return df
    
    
    def _add_system_stress_features(self, df):
        """Enhanced system stress indicators including operating reserves"""
        
        stress_scores = []
        
        # Existing stress indicators
        if 'PJM_DA_SYNC_RESERVE' in df.columns:
            reserve_high = df['PJM_DA_SYNC_RESERVE'] > df['PJM_DA_SYNC_RESERVE'].rolling(168, min_periods=1).quantile(0.9)
            stress_scores.append(reserve_high.astype(int))
        
        if 'capacity_stress' in df.columns:
            stress_scores.append(df['capacity_stress'])
        
        if 'weather_stress_score' in df.columns:
            stress_scores.append(df['weather_stress_score'])
        
        # Add operating reserve stress with higher weight
        if 'reserve_stress_score' in df.columns:
            stress_scores.append(df['reserve_stress_score'] * 1.5)  # Weight this more heavily
        
        if 'extreme_reserve_stress' in df.columns:
            stress_scores.append(df['extreme_reserve_stress'] * 2)  # Strong spike indicator
        
        if stress_scores:
            df['system_stress_score'] = sum(stress_scores) / len(stress_scores)
            
            # Create tiered stress levels
            df['stress_level_low'] = (df['system_stress_score'] < 0.3).astype(int)
            df['stress_level_medium'] = ((df['system_stress_score'] >= 0.3) & (df['system_stress_score'] < 0.6)).astype(int)
            df['stress_level_high'] = ((df['system_stress_score'] >= 0.6) & (df['system_stress_score'] < 0.8)).astype(int)
            df['stress_level_extreme'] = (df['system_stress_score'] >= 0.8).astype(int)
        else:
            df['system_stress_score'] = 0
        
        return df
    
    def _add_operating_reserve_features(self, df):
            """
            Add operating reserve features with proper handling of edge cases
            """
            
            # Get operating reserve columns
            os_reserve_cols = [col for col in df.columns if 'OS_OPERATING_RESERVE' in col or 'OS_CALC_RESERVE' in col]
            
            if not os_reserve_cols:
                print("Warning: No operating reserve data found")
                return df
            
            # 1. Lag operating reserve data
            for col in os_reserve_cols:
                # Check for invalid values in source data
                if df[col].notna().any():
                    # Replace infinite/extreme values before lagging
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].clip(lower=-1e6, upper=1e6)
                    
                    # Create lags
                    df[f'{col}_lag24'] = df[col].shift(24)
                    df[f'{col}_lag48'] = df[col].shift(48)
                    df[f'{col}_lag168'] = df[col].shift(168)
                    
                    # Change calculations with safety
                    df[f'{col}_change_24h'] = df[col] - df[col].shift(24)
                    df[f'{col}_change_pct_24h'] = self._safe_divide(
                        df[f'{col}_change_24h'],
                        df[col].shift(24).abs() + 1,
                        fill_value=0
                    ) * 100
                    
                    # Cap percentage changes
                    df[f'{col}_change_pct_24h'] = df[f'{col}_change_pct_24h'].clip(-200, 200)
            
            # 2. Reserve adequacy metrics
            if 'OS_OPERATING_RESERVE_RTO_lag24' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
                # Safe division for reserve margin
                df['reserve_margin_pct'] = self._safe_divide(
                    df['OS_OPERATING_RESERVE_RTO_lag24'],
                    df['PJM_RTO_LOAD_FORECAST_DA'],
                    fill_value=10
                ) * 100
                
                # Cap reserve margin to reasonable range
                df['reserve_margin_pct'] = df['reserve_margin_pct'].clip(0, 100)
                
                # Low reserve warnings
                df['low_reserve_warning'] = (df['reserve_margin_pct'] < 7).astype(int)
                df['critical_reserve_warning'] = (df['reserve_margin_pct'] < 5).astype(int)
                
                # Reserve percentiles with minimum window
                if len(df) > 24:
                    df['reserve_percentile_30d'] = df['OS_OPERATING_RESERVE_RTO_lag24'].rolling(
                        min(720, len(df)), min_periods=24
                    ).rank(pct=True)
                    df['reserve_below_10pct'] = (df['reserve_percentile_30d'] < 0.1).astype(int)
                else:
                    df['reserve_percentile_30d'] = 0.5
                    df['reserve_below_10pct'] = 0
                
                # Sudden reserve drops with NaN handling
                reserve_diff = df['OS_OPERATING_RESERVE_RTO_lag24'].diff(24)
                df['reserve_rapid_decline'] = (reserve_diff < -1000).fillna(False).astype(int)
            
            # 3. Regional reserve imbalances
            regional_reserve_cols = [col for col in df.columns if 'OS_CALC_RESERVE' in col and 'lag24' in col and 'RTO' not in col]
            
            if len(regional_reserve_cols) > 1:
                # Clean regional reserve data
                for col in regional_reserve_cols:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].clip(lower=0, upper=1e6)
                
                # Total regional reserves
                df['total_regional_reserves_lag24'] = df[regional_reserve_cols].sum(axis=1, skipna=True)
                
                # Safe calculation of imbalance coefficient
                regional_mean = df[regional_reserve_cols].mean(axis=1)
                regional_std = df[regional_reserve_cols].std(axis=1)
                df['reserve_imbalance_coefficient'] = self._safe_divide(
                    regional_std,
                    regional_mean,
                    fill_value=0
                )
                df['reserve_imbalance_coefficient'] = df['reserve_imbalance_coefficient'].clip(0, 2)
                
                # Zone stress indicators
                for col in regional_reserve_cols:
                    zone_name = col.replace('OS_CALC_RESERVE_', '').replace('_lag24', '')
                    if df[col].notna().sum() > 24:  # Only if we have enough data
                        zone_percentile = df[col].rolling(
                            min(720, len(df)), min_periods=24
                        ).rank(pct=True)
                        df[f'{zone_name}_reserve_stress'] = (zone_percentile < 0.2).fillna(0).astype(int)
                    else:
                        df[f'{zone_name}_reserve_stress'] = 0
                
                # East vs West balance
                eastern_zones = ['MIDATLANTIC', 'DOMINION', 'DUQUESNE', 'AP']
                western_zones = ['COMED', 'AEP', 'DAYTON', 'DEOK']
                
                eastern_reserve_cols = [col for col in regional_reserve_cols if any(z in col for z in eastern_zones)]
                western_reserve_cols = [col for col in regional_reserve_cols if any(z in col for z in western_zones)]
                
                if eastern_reserve_cols:
                    df['eastern_reserves_lag24'] = df[eastern_reserve_cols].sum(axis=1, skipna=True)
                if western_reserve_cols:
                    df['western_reserves_lag24'] = df[western_reserve_cols].sum(axis=1, skipna=True)
                if eastern_reserve_cols and western_reserve_cols:
                    df['east_west_reserve_ratio'] = self._safe_divide(
                        df['eastern_reserves_lag24'],
                        df['western_reserves_lag24'],
                        fill_value=1
                    )
                    df['east_west_reserve_ratio'] = df['east_west_reserve_ratio'].clip(0.1, 10)
            
            # 4. Reserve-to-generation ratio
            gen_cols = ['PJM_GENERATION_COAL', 'PJM_GENERATION_GAS', 'PJM_GENERATION_NUCLEAR']
            available_gen = [col for col in gen_cols if col in df.columns]
            
            if available_gen and 'OS_OPERATING_RESERVE_RTO_lag24' in df.columns:
                df['total_thermal_generation'] = df[available_gen].sum(axis=1, skipna=True)
                df['reserve_to_generation_ratio'] = self._safe_divide(
                    df['OS_OPERATING_RESERVE_RTO_lag24'],
                    df['total_thermal_generation'],
                    fill_value=0.2
                )
                df['reserve_to_generation_ratio'] = df['reserve_to_generation_ratio'].clip(0, 1)
                df['tight_reserve_generation'] = (df['reserve_to_generation_ratio'] < 0.15).astype(int)
            
            # 5. Composite stress score
            stress_indicators = []
            stress_weights = []
            
            if 'low_reserve_warning' in df.columns:
                stress_indicators.append(df['low_reserve_warning'].fillna(0))
                stress_weights.append(1)
            if 'critical_reserve_warning' in df.columns:
                stress_indicators.append(df['critical_reserve_warning'].fillna(0))
                stress_weights.append(2)
            if 'reserve_below_10pct' in df.columns:
                stress_indicators.append(df['reserve_below_10pct'].fillna(0))
                stress_weights.append(1)
            if 'reserve_rapid_decline' in df.columns:
                stress_indicators.append(df['reserve_rapid_decline'].fillna(0))
                stress_weights.append(1.5)
            if 'tight_reserve_generation' in df.columns:
                stress_indicators.append(df['tight_reserve_generation'].fillna(0))
                stress_weights.append(1)
            
            if stress_indicators:
                # Weighted average
                weighted_sum = sum(ind * w for ind, w in zip(stress_indicators, stress_weights))
                total_weight = sum(stress_weights)
                df['reserve_stress_score'] = self._safe_divide(weighted_sum, total_weight, fill_value=0)
                df['reserve_stress_score'] = df['reserve_stress_score'].clip(0, 1)
                df['extreme_reserve_stress'] = (df['reserve_stress_score'] > 0.6).astype(int)
            
            # 6. Volatility features with safety
            if 'OS_OPERATING_RESERVE_RTO_lag24' in df.columns:
                if len(df) > 24:
                    df['reserve_volatility_24h'] = df['OS_OPERATING_RESERVE_RTO_lag24'].rolling(
                        24, min_periods=12
                    ).std().fillna(0)
                    df['reserve_volatility_168h'] = df['OS_OPERATING_RESERVE_RTO_lag24'].rolling(
                        min(168, len(df)), min_periods=24
                    ).std().fillna(0)
                    
                    # Cap volatility
                    df['reserve_volatility_24h'] = df['reserve_volatility_24h'].clip(0, 10000)
                    df['reserve_volatility_168h'] = df['reserve_volatility_168h'].clip(0, 10000)
                    
                    if len(df) > 720:
                        volatility_90th = df['reserve_volatility_24h'].rolling(720, min_periods=24).quantile(0.9)
                        df['high_reserve_volatility'] = (df['reserve_volatility_24h'] > volatility_90th).fillna(0).astype(int)
                    else:
                        df['high_reserve_volatility'] = 0
            
            print("Added operating reserve features")
        
            return df
    
    def _add_capacity_adequacy_features(self, df):
        """
        Add capacity adequacy features using morning-of capacity data
        Must be lagged appropriately for next-day price prediction
        """
        
        # 1. Create lagged capacity features (24-48 hour lags minimum)
        capacity_cols = [col for col in df.columns if any(x in col for x in 
                        ['OS_TOTAL_SCHED_CAP', 'OS_UNSCHED_STM_CAP', 'INTERNAL_SCED_CAP'])]
        
        if not capacity_cols:
            print("Warning: No capacity data found")
            return df
        
        print(f"  Processing {len(capacity_cols)} capacity columns...")
        
        for col in capacity_cols:
            # Clean source data
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].clip(lower=0, upper=1e6)
            
            # Create multiple lags for prediction
            df[f'{col}_lag24'] = df[col].shift(24)  # Yesterday
            df[f'{col}_lag48'] = df[col].shift(48)  # 2 days ago
            df[f'{col}_lag168'] = df[col].shift(168)  # Week ago
            
            # Week-over-week changes
            df[f'{col}_wow_change'] = df[col] - df[col].shift(168)
            df[f'{col}_wow_change_pct'] = self._safe_divide(
                df[f'{col}_wow_change'],
                df[col].shift(168),
                fill_value=0
            ) * 100
        
        # 2. Total available capacity calculation
        if 'OS_TOTAL_SCHED_CAP_RTO_lag24' in df.columns:
            # Total capacity = scheduled + unscheduled steam
            if 'OS_UNSCHED_STM_CAP_RTO_lag24' in df.columns:
                df['total_available_capacity_lag24'] = (
                    df['OS_TOTAL_SCHED_CAP_RTO_lag24'].fillna(0) + 
                    df['OS_UNSCHED_STM_CAP_RTO_lag24'].fillna(0)
                )
            else:
                df['total_available_capacity_lag24'] = df['OS_TOTAL_SCHED_CAP_RTO_lag24']
            
            # Capacity vs load (true reserve margin)
            if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
                df['capacity_margin'] = self._safe_divide(
                    df['total_available_capacity_lag24'] - df['PJM_RTO_LOAD_FORECAST_DA'],
                    df['PJM_RTO_LOAD_FORECAST_DA'],
                    fill_value=0.15
                )
                df['capacity_margin'] = df['capacity_margin'].clip(-0.5, 1)
                
                # Capacity utilization
                df['capacity_utilization'] = self._safe_divide(
                    df['PJM_RTO_LOAD_FORECAST_DA'],
                    df['total_available_capacity_lag24'],
                    fill_value=0.7
                )
                df['capacity_utilization'] = df['capacity_utilization'].clip(0, 1)
                
                # Tight capacity indicators
                df['tight_capacity'] = (df['capacity_margin'] < 0.15).astype(int)
                df['very_tight_capacity'] = (df['capacity_margin'] < 0.10).astype(int)
                df['critical_capacity'] = (df['capacity_margin'] < 0.05).astype(int)
        
        # 3. Unscheduled steam capacity availability
        if 'OS_UNSCHED_STM_CAP_RTO_lag24' in df.columns:
            # Unscheduled capacity as buffer
            if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
                df['unsched_capacity_ratio'] = self._safe_divide(
                    df['OS_UNSCHED_STM_CAP_RTO_lag24'],
                    df['PJM_RTO_LOAD_FORECAST_DA'],
                    fill_value=0.05
                )
                df['unsched_capacity_ratio'] = df['unsched_capacity_ratio'].clip(0, 0.5)
                
                # Low unscheduled capacity warning
                df['low_unsched_capacity'] = (df['unsched_capacity_ratio'] < 0.03).astype(int)
        
        # 4. Regional capacity distribution
        regional_sched_cols = [col for col in df.columns if 'INTERNAL_SCED_CAP' in col and 
                               'lag24' in col and 'RTO' not in col]
        unsched_lag_cols = [col for col in df.columns if 'OS_UNSCHED_STM_CAP' in col and 'lag24' in col]
        regional_unsched_cols = [col for col in unsched_lag_cols if 'RTO' not in col]
        
        # USE regional_sched_cols here for internal scheduled capacity analysis
        if regional_sched_cols:
            # Regional capacity concentration
            for col in regional_sched_cols:
                zone = col.replace('INTERNAL_SCED_CAP_', '').replace('_lag24', '')
                if 'INTERNAL_SCED_CAP_RTO_lag24' in df.columns:
                    df[f'{zone}_capacity_share'] = self._safe_divide(
                        df[col],
                        df['INTERNAL_SCED_CAP_RTO_lag24'],
                        fill_value=0
                    )
                    df[f'{zone}_capacity_share'] = df[f'{zone}_capacity_share'].clip(0, 1)
            
            # East vs West capacity balance - now using regional_sched_cols
            eastern_zones = ['MIDATLANTIC', 'DOMINION']
            western_zones = ['COMED', 'AEP']
            
            eastern_cap_cols = [col for col in regional_sched_cols if any(z in col for z in eastern_zones)]
            western_cap_cols = [col for col in regional_sched_cols if any(z in col for z in western_zones)]
            
            if eastern_cap_cols:
                df['eastern_capacity_lag24'] = df[eastern_cap_cols].sum(axis=1, skipna=True)
            if western_cap_cols:
                df['western_capacity_lag24'] = df[western_cap_cols].sum(axis=1, skipna=True)
            if eastern_cap_cols and western_cap_cols:
                df['east_west_capacity_ratio'] = self._safe_divide(
                    df['eastern_capacity_lag24'],
                    df['western_capacity_lag24'],
                    fill_value=1
                )
                df['east_west_capacity_ratio'] = df['east_west_capacity_ratio'].clip(0.5, 2)
        
        # Continue with unscheduled capacity analysis
        if unsched_lag_cols:
            # Total unscheduled capacity across zones
            df['total_unsched_capacity_lag24'] = df[unsched_lag_cols].sum(axis=1, skipna=True)
            
            # Regional unscheduled capacity analysis
            if regional_unsched_cols:
                # Regional total (excluding RTO)
                df['regional_unsched_total_lag24'] = df[regional_unsched_cols].sum(axis=1, skipna=True)
                
                # Distribution metrics
                df['unsched_capacity_std'] = df[regional_unsched_cols].std(axis=1)
                df['unsched_capacity_concentration'] = self._safe_divide(
                    df['unsched_capacity_std'],
                    df[regional_unsched_cols].mean(axis=1),
                    fill_value=0
                )
                
                # Which zones have most unscheduled capacity
                df['max_unsched_zone_capacity'] = df[regional_unsched_cols].max(axis=1)
                df['min_unsched_zone_capacity'] = df[regional_unsched_cols].min(axis=1)

        
        # 5. Capacity stress score
        stress_components = []
        
        if 'tight_capacity' in df.columns:
            stress_components.append(df['tight_capacity'] * 1)
        if 'very_tight_capacity' in df.columns:
            stress_components.append(df['very_tight_capacity'] * 2)
        if 'critical_capacity' in df.columns:
            stress_components.append(df['critical_capacity'] * 3)
        if 'low_unsched_capacity' in df.columns:
            stress_components.append(df['low_unsched_capacity'] * 1.5)
        
        if stress_components:
            df['capacity_stress_score'] = sum(stress_components) / len(stress_components)
            df['capacity_stress_score'] = df['capacity_stress_score'].clip(0, 1)
            df['extreme_capacity_stress'] = (df['capacity_stress_score'] > 0.7).astype(int)
        
        print("Added capacity features")
        
        return df
    
    
    def _add_enhanced_system_stress_features(self, df):
        """Enhanced system stress combining reserves, capacity, and other factors"""
        
        stress_scores = []
        weights = []
        
        # Capacity stress (highest weight - most direct impact)
        if 'capacity_stress_score' in df.columns:
            stress_scores.append(df['capacity_stress_score'].fillna(0))
            weights.append(2.0)
        
        # Reserve stress
        if 'reserve_stress_score' in df.columns:
            stress_scores.append(df['reserve_stress_score'].fillna(0))
            weights.append(1.5)
        
        # Weather stress
        if 'weather_stress_score' in df.columns:
            stress_scores.append(df['weather_stress_score'].fillna(0))
            weights.append(1.0)
        
        # Combined regional stress
        if 'capacity_margin' in df.columns and 'reserve_margin_pct' in df.columns:
            # Critical: both capacity AND reserves are tight
            combined_tightness = (
                (df['capacity_margin'] < 0.15).astype(int) * 
                (df['reserve_margin_pct'] < 7).astype(int)
            )
            stress_scores.append(combined_tightness)
            weights.append(3.0)  # Very high weight for this combination
        
        if stress_scores:
            weighted_sum = sum(s * w for s, w in zip(stress_scores, weights))
            total_weight = sum(weights)
            df['system_stress_score'] = (weighted_sum / total_weight).clip(0, 1)
            
            # Binary indicators for model
            df['high_system_stress'] = (df['system_stress_score'] > 0.7).astype(int)
            df['extreme_system_stress'] = (df['system_stress_score'] > 0.85).astype(int)
        
        return df
    
    def _add_load_shape_features(self, df):
        """Load shape and daily patterns"""
        if 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            # Daily peak timing
            daily_peak = df.groupby(df.index.date)['PJM_RTO_LOAD_FORECAST_DA'].transform('max')
            df['load_vs_daily_peak'] = self._safe_divide(df['PJM_RTO_LOAD_FORECAST_DA'], daily_peak, 0.5)
            df['is_daily_peak_hour'] = (df['PJM_RTO_LOAD_FORECAST_DA'] == daily_peak).astype(int)
            
            # Load factor
            daily_mean = df.groupby(df.index.date)['PJM_RTO_LOAD_FORECAST_DA'].transform('mean')
            df['daily_load_factor'] = self._safe_divide(daily_mean, daily_peak, 0.5)
        
        return df
    
    def _add_regional_features(self, df):
        """Regional load and capacity imbalances"""
        
        # Add MidAtlantic if available
        if 'PJM_MIDATLANTIC_LOAD_FORECAST_DA' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['midatlantic_load_share_forecast'] = self._safe_divide(
                df['PJM_MIDATLANTIC_LOAD_FORECAST_DA'], 
                df['PJM_RTO_LOAD_FORECAST_DA'], 
                0
            )
        
        # Combined regional stress
        if 'NYISO_LOAD_FORECAST_DA' in df.columns and 'MISO_LOAD_FORECAST_DA' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['total_regional_load'] = (
                df['PJM_RTO_LOAD_FORECAST_DA'].fillna(0) + 
                df['NYISO_LOAD_FORECAST_DA'].fillna(0) + 
                df['MISO_LOAD_FORECAST_DA'].fillna(0)
            )
            df['regional_load_concentration'] = self._safe_divide(
                df['PJM_RTO_LOAD_FORECAST_DA'], 
                df['total_regional_load'], 
                0.33
            )
        
        return df
    
    def _add_cross_feature_interactions(self, df):
        """Create interaction features between key variables"""
        
        # Temperature-load interactions
        if 'temp_avg' in df.columns and 'PJM_RTO_LOAD_FORECAST_DA' in df.columns:
            df['temp_load_interaction'] = df['temp_avg'] * df['PJM_RTO_LOAD_FORECAST_DA'] / 1e6
        
        # Capacity-weather stress
        if 'capacity_stress_score' in df.columns and 'weather_stress_score' in df.columns:
            df['capacity_weather_stress'] = df['capacity_stress_score'] * df['weather_stress_score']
        
        # Reserve-load interactions
        if 'reserve_margin_pct' in df.columns and 'high_load' in df.columns:
            df['low_reserve_high_load'] = (
                (df['reserve_margin_pct'] < 10) & (df['high_load'] == 1)
            ).astype(int)
        
        return df
    
    def _add_advanced_time_features(self, df):
        """Advanced temporal patterns"""
        
        # Day of year patterns for seasonal effects
        df['week_of_year'] = df.index.isocalendar().week
        df['is_month_start'] = (df.index.day <= 3).astype(int)
        df['is_month_end'] = (df.index.day >= 28).astype(int)
        
        # Holiday proximity - FIX THE TYPE MISMATCH
        df['days_to_holiday'] = df.index.map(
            lambda x: min([abs((x.date() - h).days) for h in self.us_holidays 
                          if abs((x.date() - h).days) < 7] or [7])
        )
        df['near_holiday'] = (df['days_to_holiday'] <= 2).astype(int)
        
        return df


# =============================================================================
# PARALLEL VOTING REGRESSOR
# =============================================================================

class ParallelVotingRegressor(VotingRegressor):
    """Parallel implementation of VotingRegressor for multi-core systems"""
    
    def fit(self, X, y, sample_weight=None):
        """Parallel fit of all estimators"""
        
        # Clone estimators
        self.estimators_ = []
        
        # Fit all models in parallel
        def fit_estimator(name, estimator, X, y, sample_weight):
            est = clone(estimator)
            if sample_weight is not None:
                return est.fit(X, y, sample_weight=sample_weight)
            return est.fit(X, y)
        
        # Use joblib for parallel execution
        self.estimators_ = Parallel(n_jobs=4, backend='threading')(
            delayed(fit_estimator)(name, est, X, y, sample_weight)
            for name, est in self.estimators
        )
        
        return self
    
    def predict(self, X):
        """Parallel prediction"""
        predictions = Parallel(n_jobs=4, backend='threading')(
            delayed(est.predict)(X) for est in self.estimators_
        )
        return np.mean(predictions, axis=0)

# =============================================================================
# OPTIMIZED INTEGRATED MODEL
# =============================================================================

class IntegratedPJMPredictor:
    """
    Optimized integrated model with multi-core support
    """
    
    def __init__(self, spike_threshold=175):
        self.spike_threshold = spike_threshold
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.feature_importance = {}
        self.stress_thresholds = {}
        
        from pjm_advanced_spike_model_mac import TwoStageSpikePrediction
        self.two_stage_spike = TwoStageSpikePrediction(
            peak_threshold=125,
            offpeak_threshold=75
        )
        
    def build_ensemble(self):
        """Build optimized ensemble for multi-core execution"""
        
        # XGBoost - optimized for Ryzen
        xgb_model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1,
            min_child_weight=5,
            random_state=42,
            n_jobs=N_CORES,
            tree_method='hist',
        )
        
        # LightGBM - optimized for multi-core
        lgb_model = lgb.LGBMRegressor(
            n_estimators=800,
            max_depth=8,  # Maybe reduce to 6
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=31,
            min_child_weight=5,
            min_data_in_leaf=20,  # Add this
            reg_alpha=0.1,  # Increase from default
            reg_lambda=1.0,  # Increase from default
            random_state=42,
            n_jobs=N_CORES,
            device_type='cpu',
            force_col_wise=True,
            boosting_type='gbdt'
        )
        
        # Random Forest - parallel trees
        rf_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=N_CORES,
            max_features='sqrt'
        )
        
        # Extra Trees - parallel trees
        et_model = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=N_CORES,
            max_features='sqrt',
            bootstrap=False
        )
        
        # Use parallel voting regressor
        ensemble = ParallelVotingRegressor([
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model),
            ('random_forest', rf_model),
            ('extratrees', et_model)
        ])
        
        return ensemble
    
    def evaluate_with_predictions(self, X_test, y_test, predictions, spike_probs):
        """Evaluation using pre-computed predictions"""
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        
        # Peak/off-peak
        is_peak = ((X_test.index.hour >= 7) & (X_test.index.hour < 23) & 
                   (X_test.index.dayofweek < 5))
        
        peak_mae = mean_absolute_error(y_test[is_peak], predictions[is_peak]) if is_peak.sum() > 0 else 0
        offpeak_mae = mean_absolute_error(y_test[~is_peak], predictions[~is_peak]) if (~is_peak).sum() > 0 else 0
        
        # Spike detection
        actual_spikes = y_test > self.spike_threshold
        pred_spikes = spike_probs > 0.5
        
        if actual_spikes.sum() > 0:
            spike_recall = (actual_spikes & pred_spikes).sum() / actual_spikes.sum()
            spike_precision = (actual_spikes & pred_spikes).sum() / max(pred_spikes.sum(), 1)
            spike_f1 = 2 * (spike_precision * spike_recall) / (spike_precision + spike_recall + 1e-10)
        else:
            spike_recall = spike_precision = spike_f1 = 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'peak_mae': peak_mae,
            'offpeak_mae': offpeak_mae,
            'spike_recall': spike_recall,
            'spike_precision': spike_precision,
            'spike_f1': spike_f1
        }
    
    def _train_threshold_adjuster(self, X_train, y_train):
        """Train dynamic threshold based on system conditions"""
        
        # Calculate percentile thresholds for different stress levels
        stress_thresholds = {}
        
        if 'system_stress_score' in X_train.columns:
            for stress_level in [0.2, 0.4, 0.6, 0.8]:
                mask = X_train['system_stress_score'] >= stress_level
                if mask.sum() > 10:
                    stress_thresholds[stress_level] = y_train[mask].quantile(0.95)
        
        self.stress_thresholds = stress_thresholds
        if stress_thresholds:
            print(f"    Dynamic thresholds: {stress_thresholds}")
    
    def _calculate_spike_feature_importance(self):
        """Calculate and display spike-specific feature importance"""
        
        if 'spike' in self.models and hasattr(self.models['spike'], 'feature_importances_'):
            # Use spike feature names if we did feature selection
            feature_names = self.spike_feature_names if self.spike_feature_names else self.feature_names
            
            spike_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.models['spike'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Show top spike predictors
            print("\n  Top 10 Spike Predictors:")
            for idx, row in spike_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            self.spike_feature_importance = spike_importance
    
    def save_model(self, filepath):
        """Save the complete model package"""
        
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'spike_feature_names': self.spike_feature_names,
            'spike_feature_indices': self.spike_feature_indices,
            'feature_importance': self.feature_importance,
            'spike_feature_importance': self.spike_feature_importance,
            'spike_threshold': self.spike_threshold,
            'stress_thresholds': self.stress_thresholds,
            'spike_critical_features': self.spike_critical_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model package"""
        
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.models = model_package['models']
        self.scalers = model_package['scalers']
        self.feature_names = model_package['feature_names']
        self.spike_feature_names = model_package.get('spike_feature_names', [])
        self.spike_feature_indices = model_package.get('spike_feature_indices', [])
        self.feature_importance = model_package.get('feature_importance', {})
        self.spike_feature_importance = model_package.get('spike_feature_importance', {})
        self.spike_threshold = model_package.get('spike_threshold', 175)
        self.stress_thresholds = model_package.get('stress_thresholds', {})
        self.spike_critical_features = model_package.get('spike_critical_features', [])
        
        print(f"Model loaded from {filepath}")
        return self
    
    def predict_with_components(self, X):
        """Generate predictions with detailed component breakdown"""
        
        # Verify no NaN values
        assert not X.isna().any().any(), "Input X contains NaN values"
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Base price predictions
        price_pred = self.models['price'].predict(X_scaled)
        
        # Use selected features for spike prediction
        if hasattr(self, 'spike_feature_indices') and self.spike_feature_indices:
            X_spike = X_scaled[:, self.spike_feature_indices]
        else:
            X_spike = X_scaled
        
        spike_prob = self.models['spike'].predict_proba(X_spike)[:, 1]
        
        # Dynamic threshold adjustment based on system stress
        if hasattr(self, 'stress_thresholds') and self.stress_thresholds and 'system_stress_score' in X.columns:
            adjusted_spike_prob = spike_prob.copy()
            
            for stress_level, threshold in self.stress_thresholds.items():
                stress_mask = X['system_stress_score'].values >= stress_level
                # Increase spike probability when stress is high
                adjusted_spike_prob[stress_mask] *= (1 + stress_level * 0.5)
            
            spike_prob = np.clip(adjusted_spike_prob, 0, 1)
        
        # Adjust predictions for high spike probability
        spike_adjustment = np.where(
            spike_prob > 0.7,  # High confidence threshold
            1 + spike_prob * 0.3,  # Up to 30% price increase
            np.where(
                spike_prob > 0.5,  # Medium confidence
                1 + spike_prob * 0.15,  # Up to 15% increase
                1.0  # No adjustment
            )
        )
        
        adjusted_pred = price_pred * spike_adjustment
        
        # Cap adjustments at reasonable levels
        adjusted_pred = np.minimum(adjusted_pred, price_pred * 2)  # Max 2x base prediction
        adjusted_pred = np.maximum(adjusted_pred, price_pred * 0.5)  # Min 0.5x (avoid negative)
        
        return {
            'base_price': price_pred,
            'spike_prob': spike_prob,
            'spike_adjustment': spike_adjustment,
            'final_price': adjusted_pred
        }

    def _create_spike_sample_weights(self, X_train):
        """Create sample weights emphasizing high-stress conditions"""
        
        weights = np.ones(len(X_train))
        
        # Weight by system stress
        if 'system_stress_score' in X_train.columns:
            stress = X_train['system_stress_score'].fillna(0).values
            weights *= (1 + stress * 3)  # Up to 4x weight for extreme stress
        
        # Extra weight for capacity stress
        if 'capacity_stress_score' in X_train.columns:
            cap_stress = X_train['capacity_stress_score'].fillna(0).values
            weights *= (1 + cap_stress * 2)
        
        # Extra weight for reserve stress
        if 'reserve_stress_score' in X_train.columns:
            res_stress = X_train['reserve_stress_score'].fillna(0).values
            weights *= (1 + res_stress * 2)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    
    def _calculate_feature_importance(self, X_train):
        """Calculate and store feature importance from price model"""
        
        # Get importance from XGBoost (first model in ensemble)
        xgb_model = self.models['price'].estimators_[0]
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def train(self, X_train, y_train, X_val, y_val, use_smote=True):
        """Optimized training with parallel execution"""
        
        print("Training integrated model (optimized)...")
        start_time = time.time()
        
        # Verify no NaN values
        assert not X_train.isna().any().any(), "X_train contains NaN values"
        assert not y_train.isna().any(), "y_train contains NaN values"
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        self.scalers['features'] = RobustScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        
        # Train price prediction ensemble
        print(f"  Training price prediction ensemble using {N_CORES} cores...")
        self.models['price'] = self.build_ensemble()
        self.models['price'].fit(X_train_scaled, y_train)
        
        # Validate ensemble performance using scaled validation data
        ensemble_val_pred = self.models['price'].predict(X_val_scaled)
        ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
        print(f"    Ensemble validation MAE: ${ensemble_val_mae:.2f}/MWh")
        
        # Train two-stage spike detector
        print("  Training two-stage spike detector...")
        sample_weights = self._create_spike_sample_weights(X_train)
        
        # Two-stage uses unscaled data (has internal scaling)
        self.two_stage_spike.fit(X_train, y_train, sample_weights)
        
        # Rest of training...
        self._train_threshold_adjuster(X_train, y_train)
        self._calculate_feature_importance(X_train)
        
        # Final evaluation
        val_predictions = self.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        
        elapsed = time.time() - start_time
        print(f"  Training complete in {elapsed:.2f} seconds")
        print(f"  Final validation MAE: ${val_mae:.2f}/MWh")
        
        return self
    
    def predict(self, X, return_spike_prob=False):
        """Optimized prediction using two-stage approach"""
        
        # Verify no NaN values
        assert not X.isna().any().any(), "Input X contains NaN values"
        
        # Use two-stage spike prediction
        prices, spike_probs = self.two_stage_spike.predict(X)
        
        if return_spike_prob:
            return prices, spike_probs
        else:
            return prices
    
    def evaluate(self, X_test, y_test):
        """Evaluation with metrics"""
        
        predictions, spike_probs = self.predict(X_test, return_spike_prob=True)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        
        # Peak/off-peak
        is_peak = ((X_test.index.hour >= 7) & (X_test.index.hour < 23) & 
                   (X_test.index.dayofweek < 5))
        
        peak_mae = mean_absolute_error(y_test[is_peak], predictions[is_peak]) if is_peak.sum() > 0 else 0
        offpeak_mae = mean_absolute_error(y_test[~is_peak], predictions[~is_peak]) if (~is_peak).sum() > 0 else 0
        
        # Spike detection
        actual_spikes = y_test > self.spike_threshold
        pred_spikes = spike_probs > 0.5
        
        if actual_spikes.sum() > 0:
            spike_recall = (actual_spikes & pred_spikes).sum() / actual_spikes.sum()
            spike_precision = (actual_spikes & pred_spikes).sum() / max(pred_spikes.sum(), 1)
            spike_f1 = 2 * (spike_precision * spike_recall) / (spike_precision + spike_recall + 1e-10)
        else:
            spike_recall = spike_precision = spike_f1 = 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'peak_mae': peak_mae,
            'offpeak_mae': offpeak_mae,
            'spike_recall': spike_recall,
            'spike_precision': spike_precision,
            'spike_f1': spike_f1
        }
    
    def predict_batch(self, X, batch_size=10000):
        """Predict in batches for large datasets"""
        
        n_samples = len(X)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_pred = self.predict(batch)
            predictions.extend(batch_pred)
        
        return np.array(predictions)

    # Add model diagnostics method:

    def get_diagnostics(self):
        """Get model diagnostics and statistics"""
        
        diagnostics = {
            'n_features': len(self.feature_names),
            'spike_threshold': self.spike_threshold,
            'n_spike_features': len(self.spike_feature_indices) if self.spike_feature_indices else 0,
            'models_trained': list(self.models.keys()),
            'stress_thresholds': self.stress_thresholds
        }
        
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            diagnostics['top_5_features'] = self.feature_importance.head(5)['feature'].tolist()
        
        if self.spike_feature_importance is not None and len(self.spike_feature_importance) > 0:
            diagnostics['top_5_spike_features'] = self.spike_feature_importance.head(5)['feature'].tolist()
        
        return diagnostics

def save_detailed_predictions(model, X_test, y_test, output_path='model_output/test_predictions.csv'):
    """
    Save detailed test predictions with statistics for peak/off-peak and hourly analysis
    """
    
    # Get predictions
    predictions, spike_probs = model.predict(X_test, return_spike_prob=True)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'datetime': X_test.index,
        'actual_price': y_test.values,
        'predicted_price': predictions,
        'error': predictions - y_test.values,
        'abs_error': np.abs(predictions - y_test.values),
        'pct_error': ((predictions - y_test.values) / y_test.values) * 100,
        'spike_probability': spike_probs,
        'predicted_spike': (spike_probs > 0.5).astype(int),
        'actual_spike': (y_test > 175).astype(int)
    })
    
    # Add temporal features
    results_df['hour'] = results_df['datetime'].dt.hour
    results_df['date'] = results_df['datetime'].dt.date
    results_df['day_of_week'] = results_df['datetime'].dt.dayofweek
    results_df['month'] = results_df['datetime'].dt.month
    
    # Define peak hours (weekdays 7 AM - 11 PM)
    results_df['is_weekend'] = results_df['day_of_week'].isin([5, 6]).astype(int)
    results_df['is_peak'] = (
        (results_df['hour'] >= 7) & 
        (results_df['hour'] < 23)
    ).astype(int)
    results_df['period_type'] = results_df['is_peak'].map({1: 'On-Peak', 0: 'Off-Peak'})
    
    # Save full predictions
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed predictions saved to: {output_path}")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_df)
    
    # Save summary to separate file
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY STATISTICS")
    print("="*70)
    print(summary_stats.to_string())
    
    # Generate hourly statistics
    hourly_stats = generate_hourly_statistics(results_df)
    hourly_path = output_path.replace('.csv', '_hourly.csv')
    hourly_stats.to_csv(hourly_path)
    print(f"\nHourly statistics saved to: {hourly_path}")
    
    return results_df, summary_stats, hourly_stats

def generate_summary_statistics(results_df):
    """Generate summary statistics for overall, peak, and off-peak periods"""
    
    def calc_stats(df, name):
        return pd.Series({
            'Period': name,
            'Count': len(df),
            'MAE': df['abs_error'].mean(),
            'RMSE': np.sqrt((df['error'] ** 2).mean()),
            'MAPE': np.abs(df['pct_error']).mean(),
            'Mean_Actual': df['actual_price'].mean(),
            'Mean_Predicted': df['predicted_price'].mean(),
            'Std_Actual': df['actual_price'].std(),
            'Std_Predicted': df['predicted_price'].std(),
            'Min_Error': df['error'].min(),
            'Max_Error': df['error'].max(),
            'Error_Std': df['error'].std(),
            'Spike_Recall': (
                (df['actual_spike'] & df['predicted_spike']).sum() / 
                max(df['actual_spike'].sum(), 1)
            ),
            'Spike_Precision': (
                (df['actual_spike'] & df['predicted_spike']).sum() / 
                max(df['predicted_spike'].sum(), 1)
            ),
            'Actual_Spikes': df['actual_spike'].sum(),
            'Predicted_Spikes': df['predicted_spike'].sum()
        })
    
    # Calculate for different periods
    overall_stats = calc_stats(results_df, 'Overall')
    peak_stats = calc_stats(results_df[results_df['is_peak'] == 1], 'On-Peak')
    offpeak_stats = calc_stats(results_df[results_df['is_peak'] == 0], 'Off-Peak')
    
    # Combine
    summary = pd.DataFrame([overall_stats, peak_stats, offpeak_stats])
    summary = summary.round(2)
    
    return summary

def generate_hourly_statistics(results_df):
    """Generate statistics for each hour of the day"""
    
    hourly_stats = []
    
    for hour in range(24):
        hour_df = results_df[results_df['hour'] == hour]
        
        if len(hour_df) > 0:
            stats1 = {
                'Hour': hour,
                'Period': 'On-Peak' if hour >= 7 and hour < 23 else 'Off-Peak',
                'Count': len(hour_df),
                'MAE': hour_df['abs_error'].mean(),
                'RMSE': np.sqrt((hour_df['error'] ** 2).mean()),
                'MAPE': np.abs(hour_df['pct_error']).mean(),
                'Mean_Actual': hour_df['actual_price'].mean(),
                'Mean_Predicted': hour_df['predicted_price'].mean(),
                'Std_Error': hour_df['error'].std(),
                'Max_Actual': hour_df['actual_price'].max(),
                'Max_Predicted': hour_df['predicted_price'].max(),
                'Spike_Hours': hour_df['actual_spike'].sum(),
                'Spike_Predicted': hour_df['predicted_spike'].sum()
            }
            hourly_stats.append(stats1)
    
    hourly_df = pd.DataFrame(hourly_stats)
    hourly_df = hourly_df.round(2)
    
    return hourly_df
# =============================================================================
# OPTIMIZED MAIN PIPELINE
# =============================================================================

def run_integrated_pipeline_optimized(connection_params=None):
    """
    Optimized pipeline with performance monitoring
    """
    
    print("="*70)
    print("OPTIMIZED PJM DA PRICE PREDICTION PIPELINE")
    print(f"System: AMD Ryzen 9 7950X3D ({N_CORES} cores)")
    print("="*70)
    
    # Monitor performance
    process = psutil.Process()
    start_time = time.time()
    
    # 1. Parallel data loading
    print("\n1. Loading data (parallel)...")
    t0 = time.time()
    pipeline = PJMDataPipeline(connection_params)
    df = pipeline.load_comprehensive_data(start_date='2018-01-01')
    print(f"   Time: {time.time() - t0:.2f}s, CPU: {process.cpu_percent()}%")
    
    # 2. Feature engineering
    print("\n2. Engineering features...")
    t0 = time.time()
    engineer = EnhancedFeatureEngineer()
    df = engineer.create_all_features(df)
    print(f"   Time: {time.time() - t0:.2f}s")
    
    # 3. Prepare for modeling
    print("\n3. Preparing for modeling...")
    
    # Remove rows with target NaN
    df = df[df['DA_LMP_WEST_HUB'].notna()]
    
    # Remove current DA LMP columns from features
    da_lmp_columns = [col for col in df.columns if 'DA_LMP' in col.upper() and 'LAG' not in col.upper()]
    # CRITICAL: Remove features that shouldn't be used for DA prediction
    # These contain same-day or future information not available at prediction time
    
    features_to_remove = [
        # DA reserves - these are determined WITH the DA prices, not before
        'PJM_DA_SYNC_RESERVE',
        'PJM_DA_PRIM_RESERVE',
        
        # Real-time loads (use forecasts instead)
        'PJM_RTO_LOAD',
        'PJM_AEP_LOAD', 
        'PJM_BGE_LOAD',
        'PJM_DOMINION_LOAD',
        'PJM_PECO_LOAD',
        'PJM_PENELEC_LOAD',
        'PJM_PSEG_LOAD',
        'PJM_COMED_LOAD',
        'PJM_MIDATLANTIC_LOAD',
        
        # Operating reserves and capacity (only use lagged versions)
        'OS_OPERATING_RESERVE_RTO',
        'OS_TOTAL_SCHED_CAP_RTO',
        'OS_UNSCHED_STM_CAP_RTO',
        'OS_UNSCHED_STM_CAP_MIDATLANTIC',
        'OS_UNSCHED_STM_CAP_DOMINION',
        'OS_UNSCHED_STM_CAP_COMED',
        'OS_UNSCHED_STM_CAP_AEP',
        'OS_UNSCHED_STM_CAP_OVEC',
        
        # Calculated reserves (only use lagged)
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
        
        # Internal scheduled capacity (only use lagged)
        'INTERNAL_SCED_CAP_RTO',
        'INTERNAL_SCED_CAP_MIDATLANTIC',
        'INTERNAL_SCED_CAP_DOMINION',
        'INTERNAL_SCED_CAP_COMED',
        'INTERNAL_SCED_CAP_AEP',
        'INTERNAL_SCED_CAP_OVEC',
        
        # Real-time generation (these are after-the-fact)
        'PJM_GENERATION_COAL',
        'PJM_GENERATION_GAS',
        'PJM_GENERATION_NUCLEAR',
        'PJM_GENERATION_DFO',
        'PJM_GENERATION_SOLAR_RT',
        'PJM_GENERATION_WIND',  # Use wind forecast instead
        
        # Constraint flows (real-time)
        'PJM_BEDINGTON_BLACK_OAK_FLOW',
        'PJM_500X_FLOW',
        'PJM_INTERFACE_5004_5005_FLOW',
        
        # Derived features from above
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
        
        # RT LMPs (not available for DA prediction)
        'PJM_WESTERN_HUB_RT_LMP',
        'PJM_EASTERN_HUB_RT_LMP',
        
        # Congestion (if not lagged)
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
    ]
    
    # Remove these columns from features
    print(f"   Removing {len(features_to_remove)} non-predictive features")
    for col in features_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"     Removed: {col}")
    
    # Now check what we should be using instead
    print("\n   Verifying lagged features are present:")
    important_lags = [
        'OS_OPERATING_RESERVE_RTO_lag24',
        'OS_TOTAL_SCHED_CAP_RTO_lag24',
        'capacity_margin',
        'reserve_margin_pct',
        'PJM_RTO_LOAD_FORECAST_DA',  # This is OK - it's the forecast
        'NYISO_LOAD_FORECAST_DA',
        'MISO_LOAD_FORECAST_DA',
    ]
    
    for feat in important_lags:
        if feat in df.columns:
            print(f"     ✓ {feat}")
        else:
            print(f"     ✗ {feat} - MISSING")
    
    # Continue with rest of pipeline...
    feature_cols = [col for col in df.columns if col != 'DA_LMP_WEST_HUB' and col != 'is_spike']
    X = df[feature_cols]
    y = df['DA_LMP_WEST_HUB']
    
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n   Final features: {X.shape[1]}, Samples: {X.shape[0]}")

    
    # Split features and target
    feature_cols = [col for col in df.columns if col not in da_lmp_columns + ['is_spike']]
    X = df[feature_cols]
    y = df['DA_LMP_WEST_HUB']
    
    # Remove RT columns
    rt_lmp_columns = [col for col in X.columns if 'RT_LMP' in col.upper() and 'LAG' not in col.upper()]
    X = X.drop(columns=rt_lmp_columns, errors='ignore')
    
    rt_load_columns = [col for col in X.columns if col in ['NYISO_LOAD', 'MISO_LOAD']]
    X = X.drop(columns=rt_load_columns, errors='ignore')
    
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Final NaN check
    if X.isna().any().any():
        print("   Handling remaining NaN values...")
        X = X.fillna(method='ffill', limit=4).fillna(method='bfill', limit=4).fillna(0)
    
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Show feature categories
    print("\n   Feature categories:")
    weather_features = [col for col in X.columns if any(x in col.lower() for x in ['temp', 'cloud', 'precip', 'humidity', 'weather'])]
    load_features = [col for col in X.columns if 'load' in col.lower()]
    capacity_features = [col for col in X.columns if 'capacity' in col.lower()]
    reserve_features = [col for col in X.columns if 'reserve' in col.lower()]
    
    print(f"     Weather: {len(weather_features)}")
    print(f"     Load: {len(load_features)}")
    print(f"     Capacity: {len(capacity_features)}")
    print(f"     Reserve: {len(reserve_features)}")
    
    # 4. Time series split
    print("\n4. Creating time series splits...")
    X = X.sort_index()
    y = y.sort_index()
    
    test_start_date = X.index.max() - pd.Timedelta(days=100)
    val_start_date = test_start_date - pd.Timedelta(days=100)
    
    X_test = X[X.index >= test_start_date].copy()
    y_test = y[y.index >= test_start_date].copy()
    
    X_train_full = X[X.index < test_start_date].copy()
    y_train_full = y[y.index < test_start_date].copy()
    
    X_val = X_train_full[X_train_full.index >= val_start_date].copy()
    y_val = y_train_full[y_train_full.index >= val_start_date].copy()
    
    X_train = X_train_full[X_train_full.index < val_start_date].copy()
    y_train = y_train_full[y_train_full.index < val_start_date].copy()
    
    print(f"   Train: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"   Val: {len(X_val)} samples ({X_val.index.min().date()} to {X_val.index.max().date()})")
    print(f"   Test: {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
    
    
    # 5. Train BOTH models
    print("\n5. Training models...")
    
    # 5a. Train original model
    print("\n5a. Training base integrated model...")
    t0 = time.time()
    base_model = IntegratedPJMPredictor()
    base_model.train(X_train, y_train, X_val, y_val)
    print(f"   Base model training time: {time.time() - t0:.2f}s")
    
    # 5b. Train advanced model
    print("\n5b. Training advanced model with 2-stage spike detection...")
    t0 = time.time()
    
    # Import the advanced model
    from pjm_advanced_spike_model import AdvancedPJMPredictor
    
    advanced_model = AdvancedPJMPredictor(
        peak_spike_threshold=175,
        offpeak_spike_threshold=100,
        quantiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        coverage=0.9,
        n_jobs=N_CORES
    )
    advanced_model.fit(X_train, y_train, X_val, y_val)
    print(f"   Advanced model training time: {time.time() - t0:.2f}s")
    
    # 6. Evaluate BOTH models
    print("\n6. Evaluating both models...")
    
    # 6a. Base model evaluation
    print("\n6a. Base Model Results:")
    base_predictions, base_spike_probs = base_model.predict(X_test, return_spike_prob=True)
    base_metrics = base_model.evaluate(X_test, y_test)
    
    print(f"   MAE: ${base_metrics['mae']:.2f}/MWh")
    print(f"   RMSE: ${base_metrics['rmse']:.2f}/MWh")
    print(f"   Spike F1: {base_metrics['spike_f1']:.3f}")
    
    # 6b. Advanced model evaluation
    print("\n6b. Advanced Model Results:")
    advanced_predictions = advanced_model.predict_comprehensive(X_test)
    advanced_metrics = advanced_model.evaluate(X_test, y_test)
    
    print(f"   MAE (2-stage): ${advanced_metrics['point_prediction_mae']:.2f}/MWh")
    print(f"   MAE (QRA median): ${advanced_metrics['median_prediction_mae']:.2f}/MWh")
    print(f"   Conformal Coverage: {advanced_metrics['conformal_coverage']:.1%}")
    print(f"   Spike Recall: {advanced_metrics['spike_recall']:.1%}")
    print(f"   Spike Precision: {advanced_metrics['spike_precision']:.1%}")
    
    # 6c. Compare models
    print("\n6c. Model Comparison:")
    improvement = (base_metrics['mae'] - advanced_metrics['point_prediction_mae']) / base_metrics['mae'] * 100
    print(f"   MAE Improvement: {improvement:.1f}%")
    
    # 6d. Save detailed predictions for BOTH models
    print("\n6d. Saving predictions from both models...")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'datetime': X_test.index,
        'actual': y_test.values,
        'base_prediction': base_predictions,
        'base_spike_prob': base_spike_probs,
        'advanced_prediction': advanced_predictions['point_prediction'],
        'advanced_spike_prob': advanced_predictions['spike_probability'],
        'qra_median': advanced_predictions['median_prediction'],
        'conformal_lower': advanced_predictions['conformal_lower'],
        'conformal_upper': advanced_predictions['conformal_upper'],
        'q10': advanced_predictions['quantiles'][0.1],
        'q90': advanced_predictions['quantiles'][0.9]
    })
    
    # Add error columns
    comparison_df['base_error'] = comparison_df['base_prediction'] - comparison_df['actual']
    comparison_df['advanced_error'] = comparison_df['advanced_prediction'] - comparison_df['actual']
    comparison_df['base_abs_error'] = np.abs(comparison_df['base_error'])
    comparison_df['advanced_abs_error'] = np.abs(comparison_df['advanced_error'])
    
    # Save comparison
    os.makedirs('model_output', exist_ok=True)
    comparison_df.to_csv('model_output/model_comparison.csv', index=False)
    print("   Comparison saved to: model_output/model_comparison.csv")
    
    # Generate summary statistics for both
    summary_stats_base, _, _ = save_detailed_predictions(
        base_model, X_test, y_test,
        output_path='model_output/base_model_predictions.csv'
    )
    
    # Choose which model to save as primary
    if advanced_metrics['point_prediction_mae'] < base_metrics['mae']:
        print("\n✓ Advanced model performs better - saving as primary model")
        primary_model = advanced_model
        primary_metrics = advanced_metrics
    else:
        print("\n✓ Base model performs better - saving as primary model")
        primary_model = base_model
        primary_metrics = base_metrics
    
    # 6e. Create visualization-friendly output
    print("\n6e. Creating visualization outputs...")
    
    # Create results_df from the better performing model's predictions
    if advanced_metrics['point_prediction_mae'] < base_metrics['mae']:
        # Use advanced model predictions
        results_df = pd.DataFrame({
            'datetime': X_test.index,
            'actual_price': y_test.values,
            'predicted_price': advanced_predictions['point_prediction'],
            'spike_probability': advanced_predictions['spike_probability'],
            'error': advanced_predictions['point_prediction'] - y_test.values,
            'abs_error': np.abs(advanced_predictions['point_prediction'] - y_test.values),
            'hour': X_test.index.hour,
            'day': X_test.index.date,
            'dayofweek': X_test.index.dayofweek
        })
    else:
        # Use base model predictions
        results_df = pd.DataFrame({
            'datetime': X_test.index,
            'actual_price': y_test.values,
            'predicted_price': base_predictions,
            'spike_probability': base_spike_probs,
            'error': base_predictions - y_test.values,
            'abs_error': np.abs(base_predictions - y_test.values),
            'hour': X_test.index.hour,
            'day': X_test.index.date,
            'dayofweek': X_test.index.dayofweek
        })
    
    # Add peak/off-peak classification
    results_df['is_weekend'] = results_df['dayofweek'].isin([5, 6]).astype(int)
    results_df['is_peak'] = ((results_df['hour'] >= 7) & (results_df['hour'] < 23) & (results_df['is_weekend'] == 0)).astype(int)
    results_df['period_type'] = results_df['is_peak'].map({1: 'On-Peak', 0: 'Off-Peak'})
    
    # NOW create viz_df
    viz_df = results_df[['datetime', 'actual_price', 'predicted_price', 'period_type', 'hour']].copy()
    viz_df['date'] = viz_df['datetime'].dt.date
    
    # Daily averages for peak/off-peak
    daily_summary = viz_df.groupby(['date', 'period_type']).agg({
        'actual_price': 'mean',
        'predicted_price': 'mean'
    }).round(2)
    daily_summary.to_csv('model_output/daily_peak_offpeak_summary.csv')
    print("   Daily peak/off-peak summary saved to: model_output/daily_peak_offpeak_summary.csv")
    
    # Also save the full results_df
    results_df.to_csv('model_output/test_predictions_detailed.csv', index=False)
    print("   Detailed predictions saved to: model_output/test_predictions_detailed.csv")
   
    # Performance summary
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"CPU usage: {process.cpu_percent()}%")
    
    # 7. Save model
    print("\n7. Saving model...")
    os.makedirs('model_output', exist_ok=True)
    
    if base_model.feature_importance is not None and len(base_model.feature_importance) > 0:
        base_model.feature_importance.to_csv('model_output/feature_importance_base.csv', index=False)
        
        # Show top features
        print("\n" + "="*50)
        print("TOP 15 FEATURES (Base Model)")
        print("="*50)
        for idx, row in base_model.feature_importance.head(15).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Also save advanced model feature importance if available
    if hasattr(advanced_model.two_stage_spike, 'spike_features'):
        advanced_model.two_stage_spike.spike_features.to_csv('model_output/feature_importance_advanced.csv', index=False)
        
        print("\n" + "="*50)
        print("TOP SPIKE FEATURES (Advanced Model)")
        print("="*50)
        for idx, row in advanced_model.two_stage_spike.spike_features.head(20).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
                
    
    # Save model
    with open('model_output/optimized_pjm_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'engineer': engineer,
            'feature_names': model.feature_names,
            'metrics': metrics,
            'created_at': datetime.now()
        }, f)
    
    print("\nModel saved to: model_output/optimized_pjm_model.pkl")
    
    return primary_model, primary_metrics, comparison_df

if __name__ == "__main__":
    model, metrics = run_integrated_pipeline_optimized()