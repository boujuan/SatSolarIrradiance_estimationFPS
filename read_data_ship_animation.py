"""
Script to read irradiation data from ships in NetCDF format and plot an animation of the data on a map.
Credit: github.com/boujuan/ 

### USAGE: ###

1. Read Data:
   - Reads all NetCDF (.nc) files from a specified directory.
   - Gets the following variables: latitude ('lat'), longitude ('lon'), time ('time'), and short-wave radiation ('RAD_SW').
   - Time data should be in minutes since January 1, 1980 (epoch time).

2. Plot Data:
   - Visualize the animated trajectory on a map and plot the radiation over time side by side.
   
### CONFIG: ###

- folder_path: Set this variable to the directory containing your NetCDF files.
- variables: What to extract from the NetCDF files. Default is ['latitude', 'longitude', 'time', 'radiation'].
- variables_to_plot: Default is plotting 'radiation' against time and showing the trajectory on the map.

### DEPENDENCIES: ###

- pandas
- netCDF4 [conda install -c conda-forge netcdf4]
- matplotlib
- cartopy
- geopy (optional)
"""

import os
os.environ['CARTOPY_DIR'] = '/.cache/'
import pandas as pd
import netCDF4 as NC
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.animation as animation
from datetime import datetime, timedelta
# Cartography map import
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import geopy.distance

def read_netcdf(path):
    data = NC.Dataset(path, 'r')
    latitude = data.variables['lat'][:]
    longitude = data.variables['lon'][:]
    time = data.variables['time'][:]  # time in minutes since 1980
    short_wave_radiation = data.variables['RAD_SW'][:]    

    # Convert time from minutes since 1980 to normal time
    base_time = datetime(1980, 1, 1)  # Start date
    time = [base_time + timedelta(minutes=int(t)) for t in time]

    data.close()    
    return latitude, longitude, time, short_wave_radiation

def read_folder(folder_path, variables, extension='.nc'):
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            file_path = os.path.join(folder_path, file)
            latitude, longitude, time, radiation = read_netcdf(file_path)
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

def calculate_speed(lat1, lon1, lat2, lon2, time1, time2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    distance_km = geopy.distance.geodesic(coords_1, coords_2).km
    time_hours = (time2 - time1).total_seconds() / 3600
    speed_kmh = distance_km / time_hours
    # speed_knots = (distance_km / 1.852) / time_hours  # Convert km/h to knots
    return speed_kmh

def calculate_average_speed_degrees_per_hour(df):
    """
    Calculates the average speed of the ship in degrees per hour based on latitude and longitude coordinates over time.

    Args:
    df (DataFrame): The DataFrame containing the ship data with 'latitude', 'longitude', and 'time'.

    Returns:
    float: The average speed of the ship in degrees per hour.
    """
    total_degrees = 0
    total_time = 0

    for i in range(1, len(df)):
        lat1, lon1 = df['latitude'][i-1], df['longitude'][i-1]
        lat2, lon2 = df['latitude'][i], df['longitude'][i]
        time1, time2 = df['time'][i-1], df['time'][i]

        # Calculate change in position in degrees
        delta_lat = abs(lat2 - lat1)
        delta_lon = abs(lon2 - lon1)
        degrees_change = (delta_lat**2 + delta_lon**2)**0.5
        total_degrees += degrees_change

        # Calculate time difference in hours
        time_diff = (time2 - time1).total_seconds() / 3600
        total_time += time_diff

    # Calculate average speed in degrees per hour
    average_speed_degrees_per_hour = total_degrees / total_time if total_time > 0 else 0
    print(f"Average Speed: {average_speed_degrees_per_hour:.4f} degrees/hour")
    return average_speed_degrees_per_hour

def plot_data(df, variables):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[1])
    
    if 'latitude' in variables and 'longitude' in variables:
        line, = ax0.plot([], [], color='red', linewidth=2, transform=ccrs.PlateCarree(), label='Trajectory')
        coord_text = ax0.text(0.02, 0.05, '', transform=ax0.transAxes)
        time_text = ax0.text(0.98, 0.05, '', transform=ax0.transAxes, horizontalalignment='right')
        speed_text = ax0.text(0.02, 0.10, '', transform=ax0.transAxes)
        line_var, = ax1.plot([], [], label=variables[0])
        
        # Calculate the extent of the map based on the trajectory data
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # Add a small buffer to the extent
        # lat_buffer = (lat_max - lat_min) * 0.05
        # lon_buffer = (lon_max - lon_min) * 0.05
        
        # Set the extent of the map
        # ax0.set_extent([lon_min - lon_buffer, lon_max + lon_buffer, lat_min - lat_buffer, lat_max + lat_buffer], crs=ccrs.PlateCarree())
        ax0.set_extent([-150, -90, 0, 40], crs=ccrs.PlateCarree())
        
        ax0.coastlines()
        ax0.add_feature(cfeature.LAND)
        ax0.add_feature(cfeature.OCEAN)
        ax0.add_feature(cfeature.COASTLINE)
        ax0.add_feature(cfeature.BORDERS, linestyle=':')
        ax0.add_feature(cfeature.LAKES, alpha=0.5)
        ax0.add_feature(cfeature.RIVERS)
        ax0.set_title('Trajectory on Map')
        ax0.legend()
        
        variables = [var for var in variables if var not in ['latitude', 'longitude']]
    
    # Set initial limits for the variable
    ax1.set_xlim(df['time'].min(), df['time'].max())
    ax1.set_ylim(df[variables[0]].min(), df[variables[0]].max())
    
    # Custom tick labels for noon and midnight
    start_date = df['time'].min().date()
    end_date = df['time'].max().date()
    delta = timedelta(days=1)
    custom_ticks = []
    custom_labels = []
    current_date = start_date
    day_count = 1
    total_days = (end_date - start_date).days + 1
    while current_date <= end_date:
        midnight = datetime.combine(current_date, datetime.min.time()) + timedelta(days=1)
        if total_days > 10:
            if midnight < df['time'].max():
                custom_ticks.append(midnight)
                custom_labels.append(str(day_count))
        else:
            noon = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12)
            if noon < df['time'].max():
                custom_ticks.append(noon)
                custom_labels.append(f'D{day_count}N')
            if midnight < df['time'].max():
                custom_ticks.append(midnight)
                custom_labels.append(f'D{day_count}M')
        current_date += delta
        day_count += 1
    
    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels(custom_labels, rotation=45)
    ax1.set_xlabel('Time [UTC Days]')
    ax1.set_ylabel('GHI [W/m^2]')
    ax1.set_title(f'Ship GHI Time Series')
    ax1.legend()
    
    plt.tight_layout()
    
    def init():
        line.set_data([], [])
        coord_text.set_text('')
        time_text.set_text('')
        speed_text.set_text('')
        line_var.set_data([], [])
        return line, coord_text, line_var, time_text, speed_text
    
    def animate(i):
        if i > 0:
            speed = calculate_speed(df['latitude'][i-1], df['longitude'][i-1], df['latitude'][i], df['longitude'][i], df['time'][i-1], df['time'][i])
            speed_text.set_text(f"Speed: {speed:.2f} km/h")
        line.set_data(df['longitude'][:i+1], df['latitude'][:i+1])
        time_text.set_text(df['time'][i].strftime('%Y-%m-%d %H:%M:%S'))
        coord_text.set_text(f"Latitude: {df['latitude'][i]:.2f}, Longitude: {df['longitude'][i]:.2f}")
        line_var.set_data(df['time'][:i+1], df[variables[0]][:i+1])
        return line, coord_text, line_var, time_text, speed_text
        
    # Reduce the number of frames by only animating every nth frame
    step = 10  # Adjust this based on your dataset size and desired smoothness
    frames = range(0, len(df), step)

    # Use a faster writing speed and lower resolution
    ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=1)
    ani.save('animation2.mp4', writer='ffmpeg', fps=30, extra_args=['-preset', 'fast', '-crf', '22'])

    plt.show()
    
    
def save_combined_data_to_csv(df, output_path='data/processed/combined_data_ship.csv'):
    """
    Saves the combined data from the ship into a CSV file after adjusting longitude values.

    Args:
    df (DataFrame): The DataFrame containing the ship data.
    output_path (str): The path where the CSV file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Adjust longitude values by subtracting 360
    df['longitude'] = df['longitude'] - 360
    
    # Select only the required columns
    df_to_save = df[['time', 'latitude', 'longitude', 'radiation']]
    
    # Rename columns to match the required format
    df_to_save.columns = ['time', 'lat', 'lon', 'radiation']
    
    # Save to CSV
    df_to_save.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

# path = 'samos/netcdf/KAOU_20180825v10001.nc'
# latitude, longitude, time, radiation = read_netcdf(path)
# print(f"{latitude}, {longitude}")
# plot_data(time, radiation)

folder_path = 'data/samos/2017/netcdf'

variables = ['latitude', 'longitude', 'time', 'radiation']
variables_to_plot = ['radiation', 'latitude', 'longitude']

df = read_folder(folder_path, variables, '.nc')
# print(df)
print(df['time'].min())
print(df['time'].max())
print(df['latitude'].min())
print(df['latitude'].max())
print(df['longitude'].min())
print(df['longitude'].max())

print(df.head())
print()
print('time', df['time'].min() , df['time'].max())
print('latitude', df['latitude'].min() , df['latitude'].max())
print('longitude', df['longitude'].min() , df['longitude'].max())
print('radiation', df['radiation'].min() , df['radiation'].max())

average_speed_degrees_per_hour = calculate_average_speed_degrees_per_hour(df)
print(f"Calculated Average Speed: {average_speed_degrees_per_hour:.4f} degrees/hour")
save_combined_data_to_csv(df)
plot_data(df, variables_to_plot) # Plot animation of ship movement


