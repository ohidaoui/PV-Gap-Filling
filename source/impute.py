import pandas as pd
import numpy as np
from utils import *
import argparse



def prepare_data(meteo_path, pv_path):
    
    # Read the CSV files into DataFrames
    meteo_data = pd.read_csv(meteo_path)
    meteo_data['Time'] = pd.to_datetime(meteo_data['Time'], format='%Y-%m-%d %H:%M:%S')
    meteo_data.set_index('Time', inplace=True)
    meteo_data = meteo_data[['GTI', 'GHI', 'DNI', 'DHI', 'Air_Temp', 'RH']]
    meteo_data_10mn = meteo_data.resample('10min').mean()
    # All gaps in meteo data are asumed to be very small, use linear interpolation to replace missing values
    meteo_data_10mn.interpolate(method='linear', inplace=True)

    pv_data = pd.read_csv(pv_path)
    pv_data['Time'] = pd.to_datetime(pv_data['Time'], format='%Y-%m-%d %H:%M:%S')
    pv_data.set_index('Time', inplace=True)
    pv_data_10mn = pv_data.resample('10min').mean()

    return meteo_data_10mn, pv_data_10mn


def inv_transform_outputs(outputs, scaler):
    return [out * np.sqrt(scaler.var_[-1]) + scaler.mean_[-1] for out in outputs]


def lstm_impute(df):

    import torch
    from joblib import load
    
    # load model
    lstm_model = torch.jit.load(r'./saves/best_model_scripted.pt', map_location='cpu')
    lstm_model.eval()
    # load scaler
    scaler = load(r'./saves/std_scaler.bin')
    # FInd gaps in P_DC
    gaps = find_gaps(df['P_DC'])

    df_imputed = df.copy(deep=True)

    window_size = 24*6  # 24 hours
    
    # Impute remaining gaps
    with torch.no_grad():
        for gap in gaps:
            for i in range(df_imputed.loc[gap[0]:gap[1], 'P_DC'].shape[0]):
                window = df_imputed.loc[df_imputed.index < gap[0] + pd.Timedelta(minutes=i*10), 
                            ['GTI', 'GHI', 'DNI', 'DHI', 'Air_Temp', 'RH', 'P_DC']].tail(window_size).values
                if np.isnan(window).any(): continue
                window = scaler.fit_transform(window)
                window = torch.from_numpy(window).float().unsqueeze(0)
                forecast = lstm_model(window).cpu().numpy().item()
                df_imputed.loc[gap[0] + pd.Timedelta(minutes=i*10), 'P_DC'] = inv_transform_outputs([forecast], scaler)[0]
        
    return df_imputed['P_DC']


def multi_step_forecast(model, X_meteo, X_p_dc, num_forecast_steps=1):
    
    # Initialize an array to store the forecasts
    forecasts = []

    current_p_dc = X_p_dc

    # Perform multi-step forecasting
    for i in range(min(num_forecast_steps, X_meteo.shape[0])):
        
        # Prepare the input by stacking meteo data and power data
        current_features = np.hstack((X_meteo[i], current_p_dc))
        
        # Predict the next step
        next_step_forecast = model.predict(current_features.reshape(1, -1))

        # Append the forecast to the list
        forecasts.append(next_step_forecast[0])

        # Update the current features for the next iteration
        current_p_dc = np.roll(current_p_dc, 1)
        current_p_dc[0] = next_step_forecast

    return forecasts


def xgb_impute(df):
    
    from xgboost import XGBRegressor
    # load model
    xgb_model = XGBRegressor()
    xgb_model.load_model(r'./saves/opt_model_31_08_23.json')


    gaps = find_gaps(df['P_DC'])

    df_imputed = df.copy(deep=True)

    lag_order = 12 * 6  # 12 hours

    # Impute remaining gaps
    for gap in gaps:
        # Extract the meteo data that corresponds to the 'gap' time range
        meteo_values = df_imputed[['GTI', 'GHI', 'DNI', 'DHI', 'Air_Temp', 'RH']][gap[0]:gap[1]].values
        # Extract the power lagged values for the window before the start of the current 'gap'
        lagged_values = df_imputed.loc[df_imputed.index < gap[0], 'P_DC'].tail(lag_order).values
        forecast_step = meteo_values.shape[0]
        # Perform a multi-step forecast
        forecasts = multi_step_forecast(xgb_model, meteo_values, np.flip(lagged_values), forecast_step)
        df_imputed.loc[gap[0]:gap[1], 'P_DC'] = forecasts

    return df_imputed['P_DC']


def fill_gaps(meteo_data_10mn, pv_data_10mn, model):
    
    # Concatenate the meteo data and 'P_DC' column
    df = pd.concat([meteo_data_10mn, pv_data_10mn['P_DC']], axis=1)

    df['Imputation'] = np.nan
    
    p_dc_ts = df['P_DC'].copy(deep=True)
    p_dc_ts = impute_time_interval_with_const_value(p_dc_ts, start_hour=0, end_hour=4, const_val=0)
    p_dc_ts = impute_time_interval_with_const_value(p_dc_ts, start_hour=20, end_hour=23, const_val=0)

    df.loc[df['P_DC'].isna() & p_dc_ts.notna(), 'Imputation'] = 'night'
    df['P_DC'] = p_dc_ts.copy(deep=True)

    
    # Use linear interpolation to replace missing values in gaps of duration no more that 1h
    p_dc_ts = p_dc_ts.interpolate(method='linear', limit=6, limit_area='inside', inplace=False)

    df.loc[df['P_DC'].isna() & p_dc_ts.notna(), 'Imputation'] = 'linear'
    df['P_DC'] = p_dc_ts.copy(deep=True)

    # Impute remaining gaps with the given model
    p_dc_ts = xgb_impute(df) if model=='xgb' else lstm_impute(df)

    df.loc[df['P_DC'].isna() & p_dc_ts.notna(), 'Imputation'] = model
    df['P_DC'] = p_dc_ts.copy(deep=True)

    return df

    

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Gap filling for power generation time series data of PV (Photovoltaic) systems.")

    # Add arguments for paths to meteo_data, pv_data, and output file
    parser.add_argument("meteo_path", help="Path to the meteo data CSV file")
    parser.add_argument("pv_path", help="Path to the PV data CSV file")
    parser.add_argument("output_path", help="Path for the output imputed CSV file")
    parser.add_argument("model", nargs="?", default='xgb', help="Name of the model ('xgb' or 'lstm'), 'xgb' by default")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Prepare the data
    meteo_data_10mn, pv_data_10mn = prepare_data(args.meteo_path, args.pv_path)
    # '../data/meteo_data_2022_2.csv', '../data/System_117_2022.csv'

    assert args.model in ('xgb', 'lstm'), f"'xgb' or 'lstm' expected, got: {args.model}"

    # Call the fill_gaps function with the prepared data
    df_imputed = fill_gaps(meteo_data_10mn, pv_data_10mn, args.model)
    
    # Save the Imputed DataFrame to the output CSV file
    df_imputed.to_csv(args.output_path, index=True)
