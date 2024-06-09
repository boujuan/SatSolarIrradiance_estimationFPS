import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import os
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime

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

def interpolate_sat_to_ship(ship_data_df, sat_data_day_df):
    # Convert time columns to datetime for both dataframes
    ship_data_df['time'] = pd.to_datetime(ship_data_df['time'])
    sat_data_day_df['time'] = pd.to_datetime(sat_data_day_df['time'])
    
    # Convert datetime to seconds since epoch for interpolation
    epoch = pd.Timestamp('1970-01-01')
    ship_data_df['time_seconds'] = (ship_data_df['time'] - epoch).dt.total_seconds()
    sat_data_day_df['time_seconds'] = (sat_data_day_df['time'] - epoch).dt.total_seconds()
    
    # Prepare data for interpolation
    points = sat_data_day_df[['time_seconds', 'lat', 'lon']].values
    values = sat_data_day_df['ssi'].values
    xi = ship_data_df[['time_seconds', 'lat', 'lon']].values
    
    # Adding a small amount of noise to the data
    noise = np.random.normal(0, 1e-10, points.shape)
    points += noise
    
    # Perform 3D interpolation
    interpolated_ssi = griddata(points, values, xi, method='linear')
    
    # Create a new DataFrame with the interpolated values
    result_df = ship_data_df.copy()
    result_df['interpolated_ssi'] = interpolated_ssi
    
    # Rename columns for clarity
    result_df.rename(columns={'lat': 'ship_lat', 'lon': 'ship_lon', 'radiation': 'ship_radiation'}, inplace=True)
    
    # Select and order the desired columns
    result_df = result_df[['time', 'ship_lat', 'ship_lon', 'ship_radiation', 'interpolated_ssi']]
    
    return result_df

def separate_ship_data_by_day(ship_data_df):
    # Convert 'time' column to datetime if not already
    ship_data_df['time'] = pd.to_datetime(ship_data_df['time'])
    
    # Extract day from datetime
    ship_data_df['day'] = ship_data_df['time'].dt.strftime('%Y%m%d')
    
    # Group by day
    grouped_ship_data = {day: group.drop(columns='day') for day, group in ship_data_df.groupby('day')}
    
    return grouped_ship_data

def plot_comparison(interpolated_data, day):
    plt.figure(figsize=(10, 5))
    plt.plot(interpolated_data['time'], interpolated_data['ship_radiation'], label='Ship Radiation', color='blue')
    plt.plot(interpolated_data['time'], interpolated_data['interpolated_ssi'], label='Interpolated SSI', color='red')
    plt.xlabel('Time')
    plt.ylabel('Radiation')
    plt.title('Comparison of Ship Radiation and Interpolated SSI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/ssi_interp_comparison_{day}.png')
    plt.close()

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
# Separate ship data by day
ship_data_by_day = separate_ship_data_by_day(ship_data_df)
print("\nShip data==================================================>")
print(ship_data_df.head())
print()

desired_lat_range = (0, 35)
desired_lon_range = (-130, -105)

# Save_CSV = False
year = '2017'

print("=============================================================================================")
for day_dir in sat_day_dirs:
    day = os.path.basename(day_dir)
    date = datetime.strptime(f"{year}{day}", "%Y%j").strftime('%Y%m%d')
    # output_file = os.path.join(processed_dir, f'sat_data_{day}.csv')
    print(f"Processing directory: {day}")
    print("=============================================================================================")
    ship_data_day_df = ship_data_by_day[date]
    sat_data_day_df = aggregate_netcdf_to_dataframe_xarray(day_dir, desired_lat_range, desired_lon_range)
    # if Save_CSV:
    #     sat_data_day.to_csv(output_file, index=False)
    
    print(f"Satellite Data Overview for {day_dir} =========================>")
    print("First few entries of the dataset:")
    print(sat_data_day_df.head())
    print("\nDataFrame Shape:")
    print(sat_data_day_df.shape)
    print("\nDataFrame Columns:")
    print(sat_data_day_df.columns)
    
    if not sat_data_day_df.empty:
        print("\nSample Data:")
        print(sat_data_day_df.sample(5))  # Display a random sample of 5 rows from the DataFrame
    else:
        print("\nNo data available after filtering.")
    
    print("-"*80)
    
    # Interpolate satellite data to ship positions and compare radiation values
    interpolated_data = interpolate_sat_to_ship(ship_data_day_df, sat_data_day_df)
    print(f"Interpolated Data for {date}:")
    print(interpolated_data.head())
    
    output_interpolated_file = os.path.join('data', 'processed', f'interpolated_data_{date}.csv')
    interpolated_data.to_csv(output_interpolated_file, index=False)
    print(f"Interpolated data saved to {output_interpolated_file}")
    
    # Plot comparison
    plot_comparison(interpolated_data, date)
    
    # break
    
    