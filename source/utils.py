import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# Function to reduce memory usage of a DataFrame by downcasting numeric data types
def reduce_memory_usage(df: pd.DataFrame, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  
    # Calculate the memory usage before the reduction
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
  
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
          
            # Downcast integer columns based on their min and max values
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                  
            # Downcast float columns based on their min and max values
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)
                  
    # Calculate the memory usage after the reduction
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
  
    if verbose:
        # Print memory usage reduction information
        print("Memory usage decreased to {:5.2f} Mb  ({:.1f}%% reduction) \
        ".format(end_mem, 100 * (start_mem - end_mem) / start_mem))


# Function to impute missing values in a time series within a specified time interval with a constant value
def impute_time_interval_with_const_value(time_series: pd.Series, start_hour=0, end_hour=5, const_val=0):
    imputed_series = time_series.copy()

    for index, value in time_series.items():
        if pd.isnull(value) and start_hour <= index.hour <= end_hour:
            imputed_series[index] = const_val

    return imputed_series
    

# Function to find gaps (missing data ranges) in a time series and return them as pairs of start and end timestamps
def find_gaps(time_series: pd.Series):
    gaps = []
    start_range = None
    end_range = None

    for index, value in time_series.items():
        if pd.isna(value):
            if start_range is None:
                start_range = index
            end_range = index
        else:
            if start_range is not None:
                # missing_ranges.append((start_range, index - pd.Timedelta(minutes=1)))
                gaps.append((start_range, end_range))
                start_range = None
                end_range = None
    
    if start_range is not None:
        gaps.append((start_range, end_range))

    return gaps


# Function to create overlapping windows of a specified size from a time series
def create_overlapping_windows(time_series, window_size=100, overlap=10, eliminate_nan=True, as_numpy=False): 
    windows = []

    for i in range(0, len(time_series) - window_size + 1, window_size - overlap):
        window = time_series.iloc[i:i + window_size]
        if eliminate_nan and window.isna().any():
            continue
        if as_numpy: window = window.values # .astype(np.float16)
        windows.append(window)

    return windows


# Function to introduce missing values into a time series based on a given missing percentage range
def introduce_missing_values(ts, missing_percentage_range=(0.05, 0.3), random_seed=42):
    np.random.seed(random_seed)
    missing_percentage = np.random.uniform(*missing_percentage_range)
    num_missing = int(len(ts) * missing_percentage)
    missing_indices = np.random.choice(len(ts), num_missing, replace=False)
    ts_with_missing = ts.copy()
    ts_with_missing.iloc[missing_indices] = np.nan
    return ts_with_missing


# Function to introduce gaps (NaN values) into a time series with specified gap lengths and the number of gaps
def introduce_gaps(ts, gap_length, num_gaps=1, random_seed=42):
    np.random.seed(random_seed)
    ts_with_gaps = ts.copy()
    
    for _ in range(num_gaps):
        gap_start = np.random.randint(0, len(ts_with_gaps) - gap_length + 1)
        gap_end = gap_start + gap_length
        ts_with_gaps[gap_start:gap_end] = np.nan
        
    return ts_with_gaps


# Function to evaluate various basic imputation methods and return the mean squared error (MSE) for each method
def evaluate_imputation_methods(ts, ts_with_missing):

    def impute_mean(ts):
        return ts.fillna(ts.mean())

    def impute_median(ts):
        return ts.fillna(ts.median())
    
    def impute_mode(ts):
        return ts.fillna(ts.mode()[0])
    
    def impute_linear(ts):
        return ts.interpolate()
    
    def impute_spline(ts):
        return ts.interpolate(method='spline', order=3)
        
    def impute_locf(ts):
        return ts.fillna(method='ffill')
    
    def impute_nocb(ts):
        return ts.fillna(method='bfill')
        
    imputation_methods = {
        'Mean': impute_mean,
        'Median': impute_median,
        'Mode': impute_mode,
        'Linear Interpolation': impute_linear,
        # 'Spline Interpolation': impute_spline,
        'LOCF': impute_locf,
        'NOCB': impute_nocb,
    }
    
    results = {}

    # Iterate through each imputation method and compute MSE
    for method_name, impute_func in imputation_methods.items():
        imputed_ts = impute_func(ts_with_missing)
        mse = mean_squared_error(ts.values, imputed_ts.values)
        results[method_name] = mse
        
    return results


# Function to evaluate the performance of imputation methods over multiple trials with optional gap introduction, returning average MSE results
def evaluate_imputation_performance(ts_list, num_trials=10, gap_length=None):
    avg_results = {}

    for ts in ts_list:
        for _ in range(num_trials):
            if gap_length is not None:
                ts_with_missing = introduce_gaps(ts, gap_length)
            else:
                ts_with_missing = introduce_missing_values(ts)
            trial_results = evaluate_imputation_methods(ts, ts_with_missing)
            
            if not avg_results:
                avg_results = trial_results
            else:
                for method, mse in trial_results.items():
                    avg_results[method] += mse

    # Calculate the average MSE results over all trials and time series
    for method in avg_results:
        avg_results[method] /= (num_trials * len(ts_list))
        
    return avg_results
