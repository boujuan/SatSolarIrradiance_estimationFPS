import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import os
import pandas as pd
import glob
import xarray as xr
from scipy.interpolate import griddata

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame repr from wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows of the DataFrame

def aggregate_netcdf_to_dataframe_xarray(directory):
    files = glob.glob(os.path.join(directory, '*.nc'))
    datasets = [xr.open_dataset(file, chunks={"time": 1}).set_coords('time') for file in files]
    combined = xr.concat(datasets, dim='time')
    df = combined[['time', 'lon', 'lat', 'ssi']].to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S')  # Ensure format includes hour
    df = df.dropna(subset=['ssi'])  # Drop rows where 'ssi' is NaN
    return df

def resample_ship_data_to_sat_intervals(ship_df, sat_df):
    # Check if 'time' column exists
    if 'time' not in ship_df.columns:
        raise ValueError("Missing 'time' column in ship data DataFrame.")
    
    # Convert ship data timestamps to datetime if not already
    ship_df = ship_df.copy()  # Work on a copy to avoid modifying the original DataFrame
    ship_df['time'] = pd.to_datetime(ship_df['time'])
    
    # Set the index to the 'time' column for resampling
    ship_df_temp = ship_df.set_index('time')
    
    # Resample ship data to 1-hour intervals to match satellite data
    resampled_ship_df = ship_df_temp.resample('1h').mean().reset_index()
    
    # Ensure the resampled ship data has the same timestamps as satellite data
    resampled_ship_df = resampled_ship_df[resampled_ship_df['time'].isin(sat_df['time'])]
    return resampled_ship_df

def interpolate_sat_to_ship(ship_df, sat_df):
    # Create an empty list to store the interpolated DataFrames
    interpolated_dfs = []

    # Iterate over each timestamp in the ship data
    for timestamp in ship_df['time'].unique():
        # Get the ship data for the current timestamp
        ship_data = ship_df[ship_df['time'] == timestamp]
        
        # Get the satellite data for the current timestamp
        sat_data = sat_df[sat_df['time'] == timestamp]
        
        # Extract the latitude, longitude, and ssi values from the satellite data
        sat_lats = sat_data['lat'].values
        sat_lons = sat_data['lon'].values
        sat_ssi = sat_data['ssi'].values
        
        # Extract the latitude and longitude values from the ship data
        ship_lats = ship_data['lat'].values
        ship_lons = ship_data['lon'].values
        
        # Perform 2D interpolation using the griddata function from scipy
        # BUG: 3D interpolation by time
        interpolated_ssi = griddata((sat_lons, sat_lats, sat_data['time']), sat_ssi, (ship_lons, ship_lats), method='linear')
        
        # Create a new DataFrame for the current timestamp
        interpolated_df = pd.DataFrame({
            'time': timestamp,
            'ship_ssi': ship_data['radiation'].values[0],
            'sat_ssi': interpolated_ssi[0]
        }, index=[0])
        
        # Append the interpolated DataFrame to the list
        interpolated_dfs.append(interpolated_df)
    
    # Concatenate all the interpolated DataFrames into a single DataFrame
    interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
    
    return interpolated_df

satellite_dir = 'data/satellite/2017'
sat_day_dirs = [os.path.join(satellite_dir, d) for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]

ship_data_df = pd.read_csv('data/processed/combined_data_ship.csv')
print(ship_data_df.head())
print()

for day_dir in sat_day_dirs:
    print(f"Processing directory: {day_dir}")
    sat_data_day = aggregate_netcdf_to_dataframe_xarray(day_dir)
    day_name = os.path.basename(day_dir)
    print(sat_data_day.head())

    resampled_ship_data = resample_ship_data_to_sat_intervals(ship_data_df, sat_data_day)
    print(f"Resampled Ship Data for {day_name}:")
    print(resampled_ship_data.head())
    
    # Interpolate satellite data to ship positions and compare radiation values
    interpolated_data = interpolate_sat_to_ship(resampled_ship_data, sat_data_day)
    print(f"Interpolated Data for {day_name}:")
    print(interpolated_data.head())