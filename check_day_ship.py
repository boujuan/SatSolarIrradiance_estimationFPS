import netCDF4 as nc
import numpy as np
import datetime as dt
import cftime
import matplotlib.pyplot as plt
from netCDF4 import num2date, date2num
import matplotlib.dates as mdates

def plot_variables(ncfile, times_filtered):
    fig, axs = plt.subplots(5, 1, figsize=(12, 30), sharex=True)  # Five subplots, sharing x-axis

    # Group 1: Positional Data
    # Latitude on primary y-axis
    variables1 = ['lat']
    titles1 = ['Latitude']
    colors1 = ['b']
    for var, title, color in zip(variables1, titles1, colors1):
        data = filter_data(var)  # Filter data to match times_filtered
        axs[0].plot(times_filtered, data, label=title, color=color, linewidth=0.5)
    # axs[0].set_title('Positional Data Over Time')
    axs[0].set_ylabel('Degrees')
    axs[0].legend(loc='upper left')

    # Longitude on secondary y-axis
    ax_twin0 = axs[0].twinx()
    lon_data = filter_data('lon')  # Filter data to match times_filtered
    ax_twin0.plot(times_filtered, lon_data, label='Longitude', color='g', linewidth=0.5)
    ax_twin0.set_ylabel('Degrees')
    ax_twin0.legend(loc='upper right')

    # Group 2: Navigational Data
    # Course over Ground on primary y-axis
    variables2 = ['PL_CRS']
    titles2 = ['Course over Ground']
    colors2 = ['r']
    for var, title, color in zip(variables2, titles2, colors2):
        data = filter_data(var)  # Filter data to match times_filtered
        axs[1].plot(times_filtered, data, label=title, color=color, linewidth=0.5)
    # axs[1].set_title('Navigational Data Over Time')
    axs[1].set_ylabel('Degrees')
    axs[1].legend(loc='upper left')

    # Speed over Ground on secondary y-axis
    ax_twin1 = axs[1].twinx()
    spd_data = filter_data('PL_SPD')  # Filter data to match times_filtered
    ax_twin1.plot(times_filtered, spd_data, label='Speed over Ground', color='c', linewidth=0.5)
    ax_twin1.set_ylabel('Knots')
    ax_twin1.legend(loc='upper right')

    # Group 3: Meteorological Data - Wind Speed, Air Temperature, Humidity, Precipitation
    variables3 = ['PL_WSPD', 'T', 'RH', 'PRECIP']
    titles3 = ['Wind Speed', 'Air Temperature', 'Relative Humidity', 'Precipitation']
    colors3 = ['m', 'k', 'orange', 'blue']
    for var, title, color in zip(variables3, titles3, colors3):
        data = filter_data(var)  # Filter data to match times_filtered
        axs[2].plot(times_filtered, data, label=title, color=color)
    # axs[2].set_title('Meteorological Data Over Time - Atmosphere')
    axs[2].set_ylabel('Various Units')
    axs[2].legend()

    # Group 4: Meteorological Data - Pressure
    p_data = filter_data('P')  # Filter data to match times_filtered
    axs[3].plot(times_filtered, p_data, label='Pressure', color='y')
    # axs[3].set_title('Meteorological Data Over Time - Pressure')
    axs[3].set_ylabel('hPa')
    axs[3].legend(loc='upper left')

    # Group 5: Meteorological Data - Radiation
    rad_data = filter_data('RAD_SW')  # Filter data to match times_filtered
    axs[4].plot(times_filtered, rad_data, label='Shortwave Radiation', color='purple')
    # axs[4].set_title('Meteorological Data Over Time - Radiation')
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('W/m^2')
    axs[4].legend(loc='upper right')

    plt.tight_layout()
    
    
    # Extract month and day from the first entry in times_filtered
    plot_date = times_filtered[0].strftime('%m-%d')

    # Save the figure with the extracted date in the filename
    plt.savefig(f'figures/Instruments_{plot_date}.png')
    
    plt.show()

# Open the netCDF file 
ncfile = nc.Dataset(r'data\samos\2017\netcdf\KAOU_20171109v10002.nc')

# Get the time variable
time_var = ncfile.variables['time']

# Start date for the time conversion
start_date = dt.datetime(1980, 1, 1, 0, 0)

# Convert time variable from numeric (minutes since 1980-1-1 00:00 UTC) to datetime
times = [start_date + dt.timedelta(minutes=int(minute)) for minute in time_var[:]]

# Filter times and data to only include entries after 16:00
times_filtered = [time for time in times if time.time() >= dt.time(16, 0)]
indices = [i for i, time in enumerate(times) if time.time() >= dt.time(16, 0)]

# Filter each data array to match the filtered times
def filter_data(variable):
    return ncfile.variables[variable][:][indices]

# Update the plotting function call to use filtered data
plot_variables(ncfile, times_filtered)

# Print filtered values
print(f"Filtered Values:")
print(f"  Time: {times_filtered}")
print(f"  Latitude: {filter_data('lat')}")
print(f"  Longitude: {filter_data('lon')}")
print(f"  Course over Ground: {filter_data('PL_CRS')}")
print(f"  Speed over Ground: {filter_data('PL_SPD')}")
print(f"  Wind Speed: {filter_data('PL_WSPD')}")
print(f"  Pressure: {filter_data('P')}")
print(f"  Air Temperature: {filter_data('T')}")
print(f"  Relative Humidity: {filter_data('RH')}")
print(f"  Precipitation Accumulation: {filter_data('PRECIP')}")
print(f"  Shortwave Radiation: {filter_data('RAD_SW')}")

# Close the netCDF file
ncfile.close()
