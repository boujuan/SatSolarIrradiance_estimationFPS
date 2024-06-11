import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import glob
import pvlib
from pvlib.location import Location
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats
import seaborn as sns
import statsmodels.api as sm

def calculate_clear_sky_irradiance(time, lat, lon, altitude=0):
    # Create a location object
    site = Location(latitude=lat, longitude=lon, altitude=altitude)

    # Wrap the time in a DatetimeIndex
    time_index = pd.DatetimeIndex([time])

    # Get solar position for the given time
    solar_position = site.get_solarposition(times=time_index)

    # Estimate Linke turbidity (could be adjusted or made an input if data is available)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(time_index, lat, lon)

    # Calculate clear sky irradiance using the Ineichen model
    clear_sky = site.get_clearsky(time_index, solar_position=solar_position, model='ineichen', linke_turbidity=linke_turbidity)

    # Extract only the GHI value
    clear_sky_ghi = clear_sky['ghi'].iloc[0]

    return clear_sky_ghi

def read_netcdf_data(directory):
    # Get a list of all NetCDF files in the directory
    netcdf_files = glob.glob(os.path.join(directory, '*.nc'))

    # Initialize an empty list to store data from each file
    data_list = []

    # Start date for the time conversion
    start_date = dt.datetime(1980, 1, 1, 0, 0)

    # Loop through each NetCDF file
    for file in netcdf_files:
        # Open the NetCDF file
        ncfile = nc.Dataset(file)

        # Get the time variable
        time_var = ncfile.variables['time']

        # Convert time variable from numeric (minutes since 1980-1-1 00:00 UTC) to datetime
        times = [start_date + dt.timedelta(minutes=int(minute)) for minute in time_var[:]]
        times = [pd.to_datetime(time) for time in times]  # Convert to pandas datetime

        # Extract relevant variables from the NetCDF file
        lat = ncfile.variables['lat'][:]
        lon = ncfile.variables['lon'][:]
        radiation = ncfile.variables['RAD_SW'][:]
        speed = ncfile.variables['PL_SPD'][:]

        # Create a DataFrame for the current file
        file_data = pd.DataFrame({
            'time': times,
            'lat': lat,
            'lon': lon,
            'radiation': radiation,
            'speed': speed
        })

        # Filter data to include only times between 16:00 and 23:59
        file_data = file_data[(file_data['time'].dt.hour >= 16) & (file_data['time'].dt.hour <= 23)]

        # Append the filtered file data to the list
        data_list.append(file_data)

        # Close the NetCDF file
        ncfile.close()

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_data = pd.concat(data_list, ignore_index=True)

    return combined_data

def compute_clear_sky_index_and_error(data):
    # Convert longitude to the range of -180 to 180 degrees
    data['lon'] = data['lon'].apply(lambda x: x if x <= 180 else x - 360)

    # Calculate clear sky GHI for each row
    data['clear_sky_ghi'] = data.apply(lambda row: calculate_clear_sky_irradiance(row['time'], row['lat'], row['lon']), axis=1)
    print("Clear Sky GHI Values:", data['clear_sky_ghi'].head())  # Inspect the first few GHI values

    # Handle division by zero by adding a small epsilon to the denominator
    epsilon = 1e-10
    data['clear_sky_index'] = data['radiation'] / (data['clear_sky_ghi'] + epsilon)

    # Compute error between measured irradiation and clear sky model irradiation
    data['error'] = data['clear_sky_ghi'] - data['radiation']

    return data

def plot_clear_sky_index_vs_speed(data):
    # Bin the data by ship speed with a bin size of 1 knot
    speed_bins = np.arange(0, 25, 1)  # Adjust the bin range and step size as needed
    binned_data = data.groupby(pd.cut(data['speed'], speed_bins))
    clear_sky_index_means = binned_data['clear_sky_index'].mean().reset_index()
    sample_counts = binned_data.size().reset_index(name='count')

    # Filter bins where the sample count is greater than 100
    valid_bins = sample_counts[sample_counts['count'] > 100]
    clear_sky_index_means = clear_sky_index_means[clear_sky_index_means['speed'].isin(valid_bins['speed'])]
    sample_counts = valid_bins

    print("Speed Bins:", speed_bins)
    print("Sample Counts:", sample_counts)
    print("Valid Bins:", valid_bins)
    print("Clear Sky Index Means:", clear_sky_index_means)

    # Proceed with plotting only if there is data to plot
    if not clear_sky_index_means.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the clear sky index data
        ax1.bar(clear_sky_index_means['speed'].apply(lambda x: (x.left + x.right) / 2), clear_sky_index_means['clear_sky_index'], width=0.9, color='b', label='Clear Sky Index')
        ax1.set_xlabel('Ship Speed (knots)')
        ax1.set_ylabel('Clear Sky Index', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Clear Sky Index and Sample Count vs. Ship Speed')
        ax1.grid(True)

        # Create a secondary y-axis for the sample count
        ax2 = ax1.twinx()
        ax2.plot(sample_counts['speed'].apply(lambda x: (x.left + x.right) / 2), sample_counts['count'], color='r', marker='o', linestyle='-', label='Sample Count')
        ax2.set_ylabel('Sample Count', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add a legend
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        plt.show()
    else:
        print("No data available for plotting.")

def plot_error_vs_speed(data):
    # Bin the data by ship speed with a bin size of 1 knot
    speed_bins = np.arange(0, 25, 1)  # Adjust the bin range and step size as needed
    binned_data = data.groupby(pd.cut(data['speed'], speed_bins))
    error_means = binned_data['error'].mean().reset_index()
    sample_counts = binned_data.size().reset_index(name='count')

    # Filter bins where the sample count is greater than 100
    valid_bins = sample_counts[sample_counts['count'] > 100]
    error_means = error_means[error_means['speed'].isin(valid_bins['speed'])]
    sample_counts = valid_bins

    # Proceed with plotting only if there is data to plot
    if not error_means.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the error data
        ax1.bar(error_means['speed'].apply(lambda x: (x.left + x.right) / 2), error_means['error'], width=0.9, color='b', label='Error (W/m^2)')
        ax1.set_xlabel('Ship Speed (knots)')
        ax1.set_ylabel('Error (W/m^2)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Error and Sample Count vs. Ship Speed')
        ax1.grid(True)

        # Create a secondary y-axis for the sample count
        ax2 = ax1.twinx()
        ax2.plot(sample_counts['speed'].apply(lambda x: (x.left + x.right) / 2), sample_counts['count'], color='r', marker='o', linestyle='-', label='Sample Count')
        ax2.set_ylabel('Sample Count', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add a legend
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        plt.show()
    else:
        print("No data available for plotting.")
        
def plot_error_vs_speed_scatter(data):
    # Create a scatter plot of error vs ship speed with smaller points
    plt.figure(figsize=(10, 6))
    plt.scatter(data['speed'], data['error'], color='blue', alpha=0.5, s=2, label='Error (W/m^2) vs Speed')

    # Add a trend line to help visualize correlation
    z = np.polyfit(data['speed'], data['error'], 1)
    p = np.poly1d(z)
    plt.plot(data['speed'], p(data['speed']), "r--")  # Red dashed line for the trend

    plt.xlabel('Ship Speed (knots)')
    plt.ylabel('Error (W/m^2)')
    plt.title('Scatter Plot of Error vs. Ship Speed with Trend Line')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate Pearson correlation coefficient and the p-value
    correlation_coefficient, p_value = scipy.stats.pearsonr(data['speed'], data['error'])

    print("Correlation Coefficient:", correlation_coefficient)
    print("P-value:", p_value)

def plot_error_vs_speed_with_regression(data):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='speed', y='error', data=data, scatter_kws={'color': 'blue', 'alpha': 0.5, 's': 10}, line_kws={'color': 'red'})
    plt.xlabel('Ship Speed (knots)')
    plt.ylabel('Error (W/m^2)')
    plt.title('Regression Plot of Error vs. Ship Speed with Confidence Interval')
    plt.grid(True)
    plt.show()

def plot_residuals(data):
    # Fit a regression model
    X = sm.add_constant(data['speed'])  # Adds a constant term to the predictor
    model = sm.OLS(data['error'], X)
    results = model.fit()

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(data['speed'], results.resid, color='blue', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Ship Speed (knots)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()

# Set the directory containing the NetCDF files
netcdf_directory = 'data/samos/2017/netcdf'

# Read data from NetCDF files
data = read_netcdf_data(netcdf_directory)

# Compute clear sky index and error
data = compute_clear_sky_index_and_error(data)

# Plot error vs. ship speed
# plot_error_vs_speed(data)

# Plot clear sky index vs. ship speed
#plot_clear_sky_index_vs_speed(data)

# Plot error vs. ship speed scatter
plot_error_vs_speed_scatter(data)

# Plot error vs. ship speed with regression
plot_error_vs_speed_with_regression(data)

# Plot residuals
plot_residuals(data)
