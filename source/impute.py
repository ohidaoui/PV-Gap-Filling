import pandas as pd
import numpy as np
from utils import *
import argparse
from joblib import load
import torch


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


def model_impute(df):
    
    # load model
    best_model = torch.jit.load('../saves/best_model_scripted.pt')
    best_model.to('cpu')
    best_model.eval()
    # load scaler
    scaler = load('../saves/std_scaler.bin')

    gaps = find_gaps(df['P_DC'])

    df_imputed = df.copy(deep=True)

    window_size = 24*6
    
    # Impute remaining gaps
    with torch.no_grad():
        for gap in gaps:
            for i in range(df_imputed.loc[gap[0]:gap[1], 'P_DC'].shape[0]):
                window = df_imputed.loc[df_imputed.index < gap[0] + pd.Timedelta(minutes=i*10), 
                            ['GTI', 'GHI', 'DNI', 'DHI', 'Air_Temp', 'RH', 'P_DC']].tail(window_size).values
                if np.isnan(window).any(): continue
                window = scaler.fit_transform(window)
                window = torch.from_numpy(window).float().unsqueeze(0)
                forecast = best_model(window).cpu().numpy().item()
                df_imputed.loc[gap[0] + pd.Timedelta(minutes=i*10), 'P_DC'] = inv_transform_outputs([forecast], scaler)[0]

    return df_imputed['P_DC']


def fill_gaps(meteo_data_10mn, pv_data_10mn):
    
    # Concatenate the meteo data and 'P_DC' column
    df = pd.concat([meteo_data_10mn, pv_data_10mn['P_DC']], axis=1)

    df['P_DC'] = impute_time_interval_with_const_value(df['P_DC'], start_hour=0, end_hour=4, const_val=0)
    df['P_DC'] = impute_time_interval_with_const_value(df['P_DC'], start_hour=20, end_hour=23, const_val=0)
    
    # Use linear interpolation to replace missing values in gaps of duration no more that 1h
    df['P_DC'] = df['P_DC'].interpolate(method='linear', limit=6, inplace=False)

    # Impute remaining gaps with lstm model
    df['P_DC'] = model_impute(df)

    return df

    

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Gap filling for power generation time series data of PV (Photovoltaic) systems.")

    # Add arguments for paths to meteo_data, pv_data, and output file
    parser.add_argument("meteo_path", help="Path to the meteo data CSV file")
    parser.add_argument("pv_path", help="Path to the PV data CSV file")
    parser.add_argument("output_path", help="Path for the output imputed CSV file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the fill_gaps function with the provided paths
    meteo_data_10mn, pv_data_10mn = prepare_data(args.meteo_path, args.pv_path)
    # '../data/meteo_data_2022_2.csv', '../data/System_117_2022.csv'
    
    df_imputed = fill_gaps(meteo_data_10mn, pv_data_10mn)
    
    # Save the Imputed DataFrame to the output CSV file
    df_imputed.to_csv(args.output_path, index=True)
    
