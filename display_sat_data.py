import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

def read_netcdf(path):
    data = nc.Dataset(path, 'r')
    return data

def plot_solar_irradiance_heatmap(data):
    # Extract data
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    ssi = data.variables['ssi'][:] * data.variables['ssi'].scale_factor

    # Create a meshgrid for lat and lon
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Set up the map
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_extent([-150, -80, 0, 60])

    # Plot the data
    heatmap = ax.pcolormesh(lon2d, lat2d, ssi, cmap='hot', shading='auto', transform=ccrs.PlateCarree())

    # Add a colorbar
    cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('Solar Irradiance (W/m²)')

    # Set title and labels
    ax.set_title('Solar Irradiance Heatmap')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    ax.set_xticks(np.arange(-150, -80, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 60, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Show the plot
    plt.show()

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

data_sat_path = 'data_sat/20180810/20180810190000-OSISAF-RADFLX-01H-GOES16.nc'
data_sat = read_netcdf(data_sat_path)

print_nc_structure(data_sat)
plot_solar_irradiance_heatmap(data_sat)