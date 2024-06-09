import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import os
import pandas as pd
import dask.dataframe as dd
import xarray as xr
from scipy.interpolate import griddata

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame repr from wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows of the DataFrame

def aggregate_netcdf_to_dataframe_xarray(directory, desired_lat_range, desired_lon_range):
    # Use xr.open_mfdataset() to open multiple NetCDF files at once
    # Use combine='nested' and explicitly specify concat_dim
    combined = xr.open_mfdataset(
        os.path.join(directory, '*.nc'),
        combine='nested',
        concat_dim='time',  # Explicitly specify the dimension to concatenate along
        preprocess=lambda ds: ds.assign_coords(time=ds['time']) if 'time' in ds.variables else ds
    )
    
    # Ensure 'time', 'lat', 'lon' are set as coordinates
    if 'time' not in combined.coords:
        combined = combined.set_coords('time')
    if 'lon' not in combined.coords:
        combined = combined.set_coords('lon')
    if 'lat' not in combined.coords:
        combined = combined.set_coords('lat')
    
    # Select only the 'ssi' variable
    ssi_data = combined['ssi']

    # Convert the xarray DataArray to a pandas DataFrame with a MultiIndex
    ssi_df = ssi_data.to_dataframe(name='ssi').reset_index()

    # Drop rows where 'ssi' is NaN
    ssi_df = ssi_df.dropna(subset=['ssi'])

    # Filter rows based on the desired latitude and longitude range
    ssi_df = ssi_df[(ssi_df['lat'].between(*desired_lat_range)) & (ssi_df['lon'].between(*desired_lon_range))]

    return ssi_df

def interpolate_sat_to_ship(ship_df, sat_df):
    # Create an empty list to store the interpolated DataFrames
    interpolated_dfs = []

    # Iterate over each timestamp in the ship data
    for timestamp in ship_df['time'].unique():
        # Get the ship data for the current timestamp
        ship_data = ship_df[ship_df['time'] == timestamp]
        
        # Get the satellite data for the current timestamp
        sat_data = sat_df.loc[timestamp]
        
        # Extract the latitude, longitude, and ssi values from the satellite data
        sat_lats = sat_data.coords['lat'].values
        sat_lons = sat_data.coords['lon'].values
        sat_ssi = sat_data.values.flatten()
        
        # Extract the latitude and longitude values from the ship data
        ship_lats = ship_data['lat'].values
        ship_lons = ship_data['lon'].values
        
        # Perform 2D interpolation using the griddata function from scipy
        interpolated_ssi = griddata((sat_lons, sat_lats), sat_ssi, (ship_lons, ship_lats), method='linear')
        
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
processed_dir = 'data/processed'

sat_day_dirs = [os.path.join(satellite_dir, d) for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]
print("\nSatellite data structure==================================================>")
sat_example = xr.open_dataset('data/satellite/2017/307/20171103000000-OSISAF-RADFLX-01H-GOES13.nc')
print(sat_example)

print(sat_example['lat'].values)
print(sat_example['lon'].values)
# print(sat_example['ssi'].isel(time=0).plot())

ship_data_df = pd.read_csv('data/processed/combined_data_ship.csv')
print("\nShip data==================================================>")
print(ship_data_df.head())
print()

desired_lat_range = (0, 35)
desired_lon_range = (-130, -105)

# Save_CSV = False

print("=============================================================================================")
for day_dir in sat_day_dirs:
    # day = os.path.basename(day_dir)
    # output_file = os.path.join(processed_dir, f'sat_data_{day}.csv')
    print(f"Processing directory: {day}")
    print("=============================================================================================")
    sat_data_day = aggregate_netcdf_to_dataframe_xarray(day_dir, desired_lat_range, desired_lon_range)
    # if Save_CSV:
    #     sat_data_day.to_csv(output_file, index=False)
    
    print(f"Satellite Data Overview for {day_dir} =========================>")
    print("First few entries of the dataset:")
    print(sat_data_day.head())
    print("\nDataFrame Shape:")
    print(sat_data_day.shape)
    print("\nDataFrame Columns:")
    print(sat_data_day.columns)
    
    if not sat_data_day.empty:
        print("\nSample Data:")
        print(sat_data_day.sample(5))  # Display a random sample of 5 rows from the DataFrame
    else:
        print("\nNo data available after filtering.")
    
    print("-"*80)
    
    # Interpolate satellite data to ship positions and compare radiation values
    interpolated_data = interpolate_sat_to_ship(resampled_ship_data, sat_data_day)
    print(f"Interpolated Data for {day_name}:")
    print(interpolated_data.head())
    
    break
    
    