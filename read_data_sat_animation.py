import netCDF4 as nc
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

def aggregate_netcdf_to_dataframe_xarray_ship(directory):
    files = glob.glob(os.path.join(directory, '*.nc'))
    if not files:
        print(f"No NetCDF files found in directory {directory}.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

    datasets = [xr.open_dataset(file, chunks={"time": 1}).set_coords('time') for file in files]
    if not datasets:
        print("No datasets were created from the files.")
        return pd.DataFrame()  # Return an empty DataFrame if no datasets are created

    combined = xr.concat(datasets, dim='time')
    df = combined[['time', 'lon', 'lat', 'RAD_SW']].to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    return df

def aggregate_netcdf_to_dataframe_xarray(directory):
    files = glob.glob(os.path.join(directory, '*.nc'))
    datasets = [xr.open_dataset(file, chunks={"time": 1}).set_coords('time') for file in files]
    combined = xr.concat(datasets, dim='time')
    df = combined[['time', 'lon', 'lat', 'ssi']].to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
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
    cbar.set_label('GHI (W/m²)')

    print("Finalizing plot...")
    ax.set_title('Solar Irradiance Heatmap with Ship Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    ax.set_xticks(np.arange(-150, -80, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 60, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    ax.yaxis.set_tick_params(which='both', labelleft=True)

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

def animate_solar_irradiance(data_df, filename, ship_data_df=None):
    if 'time' not in data_df.columns:
        print("Column 'time' does not exist in DataFrame. Available columns:", data_df.columns)
        return  # Exit the function if 'time' column is not found

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    #ax.set_extent([-150, -80, 0, 60])
    ax.set_extent([-130, -100, 0, 35])

    # Set gridlines and labels
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_xticks(np.arange(-130, -100, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 35, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Dummy initialization with the first frame to avoid empty array issue
    initial_data = data_df[data_df['time'] == data_df['time'].iloc[0]]
    lon = initial_data['lon'].values
    lat = initial_data['lat'].values
    ssi = initial_data['ssi'].values

    # Use scatter instead of pcolormesh
    scatter = ax.scatter(lon, lat, c=ssi, cmap='hot', s=10, transform=ccrs.PlateCarree())  # s is the size of the point

    # Create a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.1, aspect=20)
    cbar.set_label('Solar Irradiance (W/m²)')

    ship_marker = ax.plot([], [], 'x', color='red', markersize=10, transform=ccrs.Geodetic())[0]
    ssi_label = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        current_data = data_df[data_df['time'] == frame]
        lon = current_data['lon'].values
        lat = current_data['lat'].values
        ssi = current_data['ssi'].values

        # Update scatter plot data
        scatter.set_offsets(np.c_[lon, lat])
        scatter.set_array(ssi)

        # Format the frame time for the title
        formatted_time = pd.to_datetime(frame).strftime('%Y-%m-%d %H:%M')
        ax.set_title('Satellite Solar Irradiance Heatmap - ' + formatted_time)

        if ship_data_df is not None:
            ship_frame_data = ship_data_df[ship_data_df['time'] == frame]
            if not ship_frame_data.empty:
                ship_lat = ship_frame_data['lat'].values[0]
                ship_lon = ship_frame_data['lon'].values[0]
                ship_ssi = ship_frame_data['RAD_SW'].values[0]
                ship_marker.set_data(ship_lon, ship_lat)
                ssi_label.set_text(f'Ship GHI: {ship_ssi:.2f} W/m²')
            else:
                ship_marker.set_data([], [])  # Clear previous data if no data for this frame
        else:
            ship_marker.set_data([], [])  # Ensure ship_marker is always defined

        return scatter, ship_marker, ssi_label

    ani = animation.FuncAnimation(fig, update, frames=pd.unique(data_df['time']), blit=True)
    ani.save(filename, writer='ffmpeg', fps=15, extra_args=['-preset', 'fast', '-crf', '22'])
    
################################

# data_sat_path = 'data_sat_2017/20171017150000-OSISAF-RADFLX-01H-GOES13.nc'
# data_sat = read_netcdf(data_sat_path)

# print_nc_structure(data_sat)

# # Assuming you have a folder path 'ship_data_folder' containing the ship's NetCDF files
# ship_data = read_folder_samos('samos_2017/netcdf', ['latitude', 'longitude'])
# ship_lat = ship_data['latitude'].values
# ship_lon = ship_data['longitude'].values

# # Assuming 'data_sat' is the satellite data already read by read_netcdf
# plot_solar_irradiance_heatmap(data_sat, ship_lat, ship_lon)

################################

base_directory = 'data/satellite/2017'
# save_daily_satellite_data_to_csv(base_directory) # TOO MUCH DATA!!!

day_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

ship_data_df = aggregate_netcdf_to_dataframe_xarray_ship('data/samos/2017/netcdf')
# ship_data_df = pd.read_csv('data/processed/combined_data_ship.csv')
ship_data_df.rename(columns={
    'time': 'time',  # Assuming 'time' is already correctly named
    'lat': 'lat',
    'lon': 'lon',
    'radiation': 'RAD_SW'
}, inplace=True)

for day_dir in day_directories:
    print(f"Processing directory: {day_dir}")
    all_data_df = aggregate_netcdf_to_dataframe_xarray(day_dir)
    day_name = os.path.basename(day_dir)  # Extracts the day part from the path
    animation_filename = f'solar_irradiance_animation_{day_name}.mp4'  # Creates a unique filename
    animate_solar_irradiance(all_data_df, animation_filename, ship_data_df)