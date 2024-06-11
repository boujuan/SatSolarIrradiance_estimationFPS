import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def plot_grid_and_trajectory(satellite_file, ship_trajectory_file, lon_range, lat_range):
    # Load satellite data using xarray
    sat_data = xr.open_dataset(satellite_file)
    lat_grid = sat_data['lat'].values
    lon_grid = sat_data['lon'].values
    
    # Calculate resolution of latitude and longitude
    lat_resolution = np.diff(lat_grid).mean()  # Assuming uniform spacing
    lon_resolution = np.diff(lon_grid).mean()  # Assuming uniform spacing
    print(f"Latitude resolution: {lat_resolution}, Longitude resolution: {lon_resolution}")

    print(f"Coordinates: {lon_grid}, {lat_grid}")

    # Filter satellite data for specified longitude and latitude ranges
    # lon_mask = (lon_grid >= -130) & (lon_grid <= -115)
    # lat_mask = (lat_grid >= 3) & (lat_grid <= 34)
    lon_mask = (lon_grid >= lon_range[0]) & (lon_grid <= lon_range[1])
    lat_mask = (lat_grid >= lat_range[0]) & (lat_grid <= lat_range[1])
    lon_grid_filtered = lon_grid[lon_mask]
    lat_grid_filtered = lat_grid[lat_mask]

    # Create 2D mesh grids for filtered lat and lon
    lon2d, lat2d = np.meshgrid(lon_grid_filtered, lat_grid_filtered)

    # Load ship data
    ship_data = pd.read_csv(ship_trajectory_file)
    ship_data['time'] = pd.to_datetime(ship_data['time'])
    ship_data.set_index('time', inplace=True)
    ship_data = ship_data.resample('1h').mean()  # Resample ship data to 1-hour segments

    # Set up the plot with Cartopy
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    ax.coastlines()

    # Set up gridlines and tick labels
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot the filtered satellite grid lines
    for i in range(lon2d.shape[1]):
        ax.plot(lon2d[:, i], lat2d[:, i], 'b-', transform=ccrs.PlateCarree(), alpha=0.5)  # Vertical lines
    for j in range(lat2d.shape[0]):
        ax.plot(lon2d[j, :], lat2d[j, :], 'b-', transform=ccrs.PlateCarree(), alpha=0.5)  # Horizontal lines

    # Plot the ship trajectory with markers
    ship_lons = ship_data['lon'].values
    ship_lats = ship_data['lat'].values
    ax.plot(ship_lons, ship_lats, 'k-', transform=ccrs.Geodetic(), label='Ship Trajectory')
    ax.scatter(ship_lons, ship_lats, color='red', marker='.', s=100, transform=ccrs.Geodetic())  # Mark each hour

    # Highlight segments crossing the grid boundaries and print their details
    print("Crossing Segments:")
    for i in range(1, len(ship_lats)):
        # Check if the segment crosses any vertical grid line
        crosses_lon = np.any((lon_grid_filtered[:-1] <= ship_lons[i-1]) & (lon_grid_filtered[1:] >= ship_lons[i])) or \
                      np.any((lon_grid_filtered[:-1] >= ship_lons[i-1]) & (lon_grid_filtered[1:] <= ship_lons[i]))
        # Check if the segment crosses any horizontal grid line
        crosses_lat = np.any((lat_grid_filtered[:-1] <= ship_lats[i-1]) & (lat_grid_filtered[1:] >= ship_lats[i])) or \
                      np.any((lat_grid_filtered[:-1] >= ship_lats[i-1]) & (lat_grid_filtered[1:] <= ship_lats[i]))
        if crosses_lat or crosses_lon:
            ax.plot(ship_lons[i-1:i+1], ship_lats[i-1:i+1], 'b-', linewidth=2, transform=ccrs.Geodetic())
            # print(f"Segment {i}: Time {ship_data.index[i-1]} to {ship_data.index[i]}, Lat {ship_lats[i-1]} to {ship_lats[i]}, Lon {ship_lons[i-1]} to {ship_lons[i]}")

    ax.legend(loc='upper right')
    plt.show()

# Example usage
satellite_file = r'data\satellite\2017\289\20171016000000-OSISAF-RADFLX-01H-GOES13.nc'
ship_trajectory_file = r'data\processed\combined_data_ship.csv'
lon_range = [-125.5, -123.5]
lat_range = [9, 11]
plot_grid_and_trajectory(satellite_file, ship_trajectory_file, lon_range, lat_range)
