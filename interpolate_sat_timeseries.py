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
import glob
from memory_profiler import profile

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame repr from wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows of the DataFrame

@profile
def aggregate_netcdf_to_dataframe_xarray(directories, desired_lat_range, desired_lon_range):
    # Collect all file paths from all directories using glob to expand wildcards
    file_paths = [glob.glob(os.path.join(dir, '*.nc')) for dir in directories]
    file_paths = [item for sublist in file_paths for item in sublist]  # Flatten the list

    def preprocess(ds):
        # Convert 'ssi' to float32 if it exists
        if 'ssi' in ds.variables:
            ds['ssi'] = ds['ssi'].astype('float32')
        # Convert integer types from int64 to int32
        for var in ds.variables:
            if ds[var].dtype == 'int64':
                ds[var] = ds[var].astype('int32')
        return ds.assign_coords(time=ds['time']) if 'time' in ds.variables else ds

    combined = xr.open_mfdataset(
        file_paths,
        engine='netcdf4',
        combine='nested',
        concat_dim='time',
        preprocess=preprocess,
        chunks={'time': 5}  # Adjust based on your dataset dimensions
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
    def convert_in_chunks(data_array, chunk_size=50):  # Smaller chunk size
        num_chunks = (len(data_array.time) // chunk_size) + 1
        df_list = []
        for i in range(num_chunks):
            chunk = data_array.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
            df = chunk.to_dataframe(name='ssi').reset_index()
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    ssi_df = convert_in_chunks(ssi_data)

    # Drop rows where 'ssi' is NaN
    ssi_df = ssi_df.dropna(subset=['ssi'])

    # Filter rows based on the desired latitude and longitude range
    ssi_df = ssi_df[(ssi_df['lat'].between(*desired_lat_range)) & (ssi_df['lon'].between(*desired_lon_range))]

    return ssi_df

def interpolate_sat_to_ship(ship_data_ts_df, sat_data_ts_df):
    # Convert time columns to datetime for both dataframes
    ship_data_ts_df['time'] = pd.to_datetime(ship_data_ts_df['time'])
    sat_data_ts_df['time'] = pd.to_datetime(sat_data_ts_df['time'])
    
    # Convert datetime to seconds since epoch for interpolation
    epoch = pd.Timestamp('1970-01-01')
    ship_data_ts_df['time_seconds'] = (ship_data_ts_df['time'] - epoch).dt.total_seconds()
    sat_data_ts_df['time_seconds'] = (sat_data_ts_df['time'] - epoch).dt.total_seconds()
    
    # Prepare data for interpolation
    points = sat_data_ts_df[['time_seconds', 'lat', 'lon']].values
    values = sat_data_ts_df['ssi'].values
    xi = ship_data_ts_df[['time_seconds', 'lat', 'lon']].values
    
    # Adding a small amount of noise to the data
    noise = np.random.normal(0, 1e-10, points.shape)
    points += noise
    
    # Convert seconds since epoch back to datetime for printing
    last_10_points_time = pd.to_datetime(points[-10:, 0], unit='s', origin='unix')
    last_10_xi_time = pd.to_datetime(xi[-10:, 0], unit='s', origin='unix')
        
    # Perform 3D interpolation
    interpolated_ssi = griddata(points, values, xi, method='linear')
    
    # Create a new DataFrame with the interpolated values
    result_df = ship_data_ts_df.copy()
    result_df['interpolated_ssi'] = interpolated_ssi
    
    # Rename columns for clarity
    result_df.rename(columns={'lat': 'ship_lat', 'lon': 'ship_lon', 'radiation': 'ship_radiation'}, inplace=True)
    
    # Select and order the desired columns
    result_df = result_df[['time', 'ship_lat', 'ship_lon', 'ship_radiation', 'interpolated_ssi']]
    
    return result_df

def plot_comparison(interpolated_data):
    plt.figure(figsize=(10, 5))
    plt.plot(interpolated_data['time'], interpolated_data['ship_radiation'], label='Ship Radiation', color='blue')
    plt.plot(interpolated_data['time'], interpolated_data['interpolated_ssi'], label='Interpolated SSI', color='red')
    plt.xlabel('Time')
    plt.ylabel('Radiation')
    plt.title('Comparison of Ship Radiation and Interpolated SSI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/ssi_interp_comparison_timeseries.png')
    plt.close()

# INFO: Define satellite and ship data directories
satellite_dir = r'data\satellite\2017'
processed_dir = r'data\processed'
desired_lat_range = (0, 35)
desired_lon_range = (-130, -105)
year = '2017'

# Instead of looping through each day directory
# Now sat_data_df contains data for the entire timeseries
sat_day_dirs = [os.path.join(satellite_dir, d) for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]
print(f"Satellite day directories: {sat_day_dirs}")
sat_data_ts_df = aggregate_netcdf_to_dataframe_xarray(sat_day_dirs, desired_lat_range, desired_lon_range)
ship_data_ts_df = pd.read_csv(r'data\processed\combined_data_ship.csv')

print("\nShip data==================================================>")
print(ship_data_ts_df.head())
print()

print("=============================================================================================")
print(f"Satellite Data Overview =========================>")
print("First few entries of the dataset:")
print(sat_data_ts_df.head())
print("\nDataFrame Shape:")
print(sat_data_ts_df.shape)
print("\nDataFrame Columns:")
print(sat_data_ts_df.columns)

if not sat_data_ts_df.empty:
    print("\nSample Data:")
    print(sat_data_ts_df.sample(5))  # Display a random sample of 5 rows from the DataFrame
else:
    print("\nNo data available after filtering.")

print("-"*80)

# Interpolate satellite data to ship positions and compare radiation values
interpolated_data = interpolate_sat_to_ship(ship_data_ts_df, sat_data_ts_df)
print(f"Interpolated Data:")
print(interpolated_data.head())

# INFO: Save interpolated data to csv
# output_interpolated_file = os.path.join('data', 'processed', f'interpolated_data_{date}.csv')
# interpolated_data.to_csv(output_interpolated_file, index=False)
# print(f"Interpolated data saved to {output_interpolated_file}")

# INFO: Plot comparison
plot_comparison(interpolated_data)