import netCDF4 as nc
# import matplotlib
# matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.animation as animation
import xarray as xr
import dask

def read_netcdf(path):
    data = nc.Dataset(path, 'r')
    return data

def read_folder_samos(folder_path, variables, extension='.nc'):
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            file_path = os.path.join(folder_path, file)
            dataset = read_netcdf(file_path)  # Get the dataset
            latitude = dataset.variables['lat'][:]
            longitude = dataset.variables['lon'][:]
            time = dataset.variables['time'][:]  # Adjust this if the variable name differs
            radiation = dataset.variables['RAD_SW'][:]  # Adjust this if the variable name differs

            data = {
                'latitude': latitude,
                'longitude': longitude,
                'time': time,
                'radiation': radiation
            }
            df = pd.DataFrame(data)
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df[variables]

def read_netcdf_with_xarray(path):
    data = xr.open_dataset(path, chunks={})
    return data

def aggregate_netcdf_to_dataframe_xarray(directory):
    files = glob.glob(os.path.join(directory, '*.nc'))
    datasets = [xr.open_dataset(file, chunks={"time": 1}).set_coords('time') for file in files]
    combined = xr.concat(datasets, dim='time')
    df = combined[['time', 'lon', 'lat', 'ssi']].to_dataframe().reset_index()
    # Convert 'time' to datetime format if it's not already
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    return df

def aggregate_netcdf_to_dataframe(directory):
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    
    # List all NetCDF files in the directory
    files = glob.glob(os.path.join(directory, '*.nc'))
    
    for file in files:
        data = nc.Dataset(file, 'r')
        
        # Extract variables
        time = data.variables['time'][:]
        lon = data.variables['lon'][:]
        lat = data.variables['lat'][:]
        ssi = data.variables['ssi'][:] * data.variables['ssi'].scale_factor
        
        # Check if time is a scalar and handle accordingly
        if time.shape == ():
            time = np.array([time])  # Make time an array with one element
        
        # Calculate the number of repetitions needed for each dimension
        num_repeats = len(lon) * len(lat)
        time_repeated = np.repeat(time, num_repeats)
        lon_repeated = np.tile(np.repeat(lon, len(lat)), len(time))
        lat_repeated = np.tile(lat, len(time) * len(lon))
        
        # Flatten the SSI array to match the dimensions of the other arrays
        ssi_flattened = ssi.flatten()
        
        # Create a DataFrame for the current file
        temp_df = pd.DataFrame({
            'Time': time_repeated,
            'Longitude': lon_repeated,
            'Latitude': lat_repeated,
            'SSI': ssi_flattened
        })
        
        # Append to the main DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)
    
    return df

def plot_solar_irradiance_heatmap(data, ship_lat=None, ship_lon=None):
    print("Extracting data...")
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    ssi = data.variables['ssi'][:] * data.variables['ssi'].scale_factor

    print("Creating meshgrid...")
    lon2d, lat2d = np.meshgrid(lon, lat)

    print("Setting up the map...")
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()

    decimation_factor = 10  # Only plot every 10th point
    print("Plotting data...")
    heatmap = ax.pcolormesh(lon2d[::decimation_factor, ::decimation_factor], lat2d[::decimation_factor, ::decimation_factor], ssi[::decimation_factor, ::decimation_factor], cmap='hot', shading='auto', transform=ccrs.PlateCarree())

    print("Adding colorbar...")
    cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('Solar Irradiance (W/m²)')

    print("Finalizing plot...")
    ax.set_title('Solar Irradiance Heatmap with Ship Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    ax.set_xticks(np.arange(-150, -80, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 60, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Plot the ship's trajectory if provided
    if ship_lat is not None and ship_lon is not None:
        ax.plot(ship_lon, ship_lat, marker='o', color='blue', markersize=5, transform=ccrs.Geodetic(), label='Ship Trajectory')
        ax.legend()

    print("Displaying plot...")
    plt.show()
    print("Done.")

def print_nc_structure(data, indent=0):
    indent_str = ' ' * indent
    # Print attributes of the current data object
    for attr_name in data.ncattrs():
        print(f"{indent_str}Attribute - {attr_name}: {getattr(data, attr_name)}")
    # Print variables and their details
    for var_name, variable in data.variables.items():
        print(f"{indent_str}Variable - {var_name}: {variable}")
        print(f"{indent_str}  dimensions: {variable.dimensions}")
        print(f"{indent_str}  size: {variable.size}")
        print(f"{indent_str}  data type: {variable.dtype}")
        for attr_name in variable.ncattrs():
            print(f"{indent_str}  {attr_name}: {variable.getncattr(attr_name)}")
    # Recursively print groups and their contents
    for group_name, group in data.groups.items():
        print(f"{indent_str}Group - {group_name}")
        print_nc_structure(group, indent + 4)

def animate_solar_irradiance(data_df, filename):
    if 'time' not in data_df.columns:
        print("Column 'time' does not exist in DataFrame. Available columns:", data_df.columns)
        return  # Exit the function if 'time' column is not found

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_extent([-150, -80, 0, 60])
    scatter = ax.scatter([], [], c=[], cmap='hot', s=10, transform=ccrs.PlateCarree())

    def update(frame):
        current_data = data_df[data_df['time'] == frame]
        scatter.set_offsets(np.c_[current_data['lon'], current_data['lat']])
        scatter.set_array(current_data['ssi'])
        ax.set_title('Solar Irradiance Heatmap - Time: ' + str(frame))
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=pd.unique(data_df['time']), blit=True)
    ani.save(filename, writer='ffmpeg', fps=30, extra_args=['-preset', 'fast', '-crf', '22'])
    
################################

data_sat_path = 'data_sat_2017/20171017150000-OSISAF-RADFLX-01H-GOES13.nc'
data_sat = read_netcdf(data_sat_path)

print_nc_structure(data_sat)

# Assuming you have a folder path 'ship_data_folder' containing the ship's NetCDF files
ship_data = read_folder_samos('samos_2017/netcdf', ['latitude', 'longitude'])
ship_lat = ship_data['latitude'].values
ship_lon = ship_data['longitude'].values

# Assuming 'data_sat' is the satellite data already read by read_netcdf
plot_solar_irradiance_heatmap(data_sat, ship_lat, ship_lon)

################################

# base_directory = 'data_sat'
# day_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# for day_dir in day_directories:
#     print(f"Processing directory: {day_dir}")
#     all_data_df = aggregate_netcdf_to_dataframe_xarray(day_dir)
#     day_name = os.path.basename(day_dir)  # Extracts the day part from the path
#     animation_filename = f'solar_irradiance_animation_{day_name}.mp4'  # Creates a unique filename
#     animate_solar_irradiance(all_data_df, animation_filename)

