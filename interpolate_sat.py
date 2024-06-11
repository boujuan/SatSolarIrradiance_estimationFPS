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
import pvlib
from pvlib.location import Location

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame repr from wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows of the DataFrame

def calculate_clear_sky_irradiance(ship_data_df, altitude=0):
    # Ensure 'time' column is in pd.Timestamp format
    ship_data_df['time'] = pd.to_datetime(ship_data_df['time'])

    # Create a temporary DataFrame to store the results
    results = []

    # Process each row using vectorized operations
    for _, row in ship_data_df.iterrows():
        time = pd.Timestamp(row['time'])
        lat = row['lat']
        lon = row['lon']

        # Wrap the time in a DatetimeIndex
        time_index = pd.DatetimeIndex([time])

        # Create a location object
        site = Location(latitude=lat, longitude=lon, altitude=altitude)

        # Get solar position for the given time
        solar_position = site.get_solarposition(times=time_index)

        # Estimate Linke turbidity (could be adjusted or made an input if data is available)
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(time_index, lat, lon)

        # Calculate clear sky irradiance using the Ineichen model
        clear_sky = site.get_clearsky(time_index, solar_position=solar_position, model='ineichen', linke_turbidity=linke_turbidity)

        # Extract only the GHI value
        ghi = clear_sky['ghi'].iloc[0]

        # Append result
        results.append(ghi)

    # Add the results as a new column to the DataFrame
    ship_data_df['clear_sky_ghi'] = results
    
    print(ship_data_df.head())

    return ship_data_df

def aggregate_netcdf_to_dataframe_xarray(directory, desired_lat_range, desired_lon_range, include_next_day_start=False):
    # Open multiple NetCDF files
    combined = xr.open_mfdataset(
        os.path.join(directory, '*.nc'),
        combine='nested',
        concat_dim='time',
        preprocess=lambda ds: ds.assign_coords(time=ds['time']) if 'time' in ds.variables else ds
    )

    # Optionally include the first timestamp from the next day
    if include_next_day_start:
        next_day_directory = os.path.join(os.path.dirname(directory), str(int(os.path.basename(directory)) + 1))
        next_day_pattern = os.path.join(next_day_directory, '*000000-OSISAF-RADFLX-01H-GOES13.nc')
        next_day_files = glob.glob(next_day_pattern)  # Use glob to find files matching the pattern
        print("next_day_files: ", next_day_files)
        if next_day_files:  # Check if any files were found
            try:
                next_day_file = xr.open_dataset(next_day_files[0])  # Open the first matching file
                print("Next day file structure: ", next_day_file)
                # Extract the time value and add it as a coordinate to the dataset
                time_value = next_day_file['time'].values
                next_day_file = next_day_file.expand_dims(time=pd.Index([time_value], name='time'))
                combined = xr.concat([combined, next_day_file], dim='time')
            except FileNotFoundError:
                print("Next day file not found. Proceeding without it.")
        else:
            print("No files match the pattern for the next day. Proceeding without it.")

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
    
    # Perform 3D interpolation using cubic method
    interpolated_ssi = griddata(points, values, xi, method='linear')
    
    # Create a new DataFrame with the interpolated values
    result_df = ship_data_df.copy()
    result_df['interpolated_ssi'] = interpolated_ssi
    
    # Rename columns for clarity
    result_df.rename(columns={'lat': 'ship_lat', 'lon': 'ship_lon', 'radiation': 'ship_radiation'}, inplace=True)
    
    # Select and order the desired columns including 'clear_sky_ghi'
    result_df = result_df[['time', 'ship_lat', 'ship_lon', 'ship_radiation', 'clear_sky_ghi', 'interpolated_ssi']]
    
    return result_df

def separate_ship_data_by_day(ship_data_df):
    # Convert 'time' column to datetime if not already
    ship_data_df['time'] = pd.to_datetime(ship_data_df['time'])
    
    # Extract day from datetime
    ship_data_df['day'] = ship_data_df['time'].dt.strftime('%Y%m%d')
    
    # Group by day
    grouped_ship_data = {day: group.drop(columns='day') for day, group in ship_data_df.groupby('day')}
    
    return grouped_ship_data

def calculate_errors(interpolated_data):
    # Calculate residual errors
    residuals = interpolated_data['interpolated_ssi'] - interpolated_data['ship_radiation']
    
    # Mean Bias Error
    mbe = residuals.mean()
    
    # Root Mean Squared Error
    rmse = np.sqrt((residuals**2).mean())
    
    errors = {
        'residuals': residuals,
        'MBE': mbe,
        'RMSE': rmse
    }
    
    print(f"Mean Bias Error: {mbe}")
    print(f"Root Mean Squared Error: {rmse}")
    
    return errors

def plot_comparison(interpolated_data, day, start_time='15:30', end_time='23:59'):
    # Convert 'time' column to datetime if not already
    interpolated_data['time'] = pd.to_datetime(interpolated_data['time'])

    # Filter data within the specified time range
    if end_time > start_time:
        mask = (interpolated_data['time'].dt.time >= pd.to_datetime(start_time).time()) & \
               (interpolated_data['time'].dt.time <= pd.to_datetime(end_time).time())
    else:  # Over midnight scenario
        mask = (interpolated_data['time'].dt.time >= pd.to_datetime(start_time).time()) | \
               (interpolated_data['time'].dt.time <= pd.to_datetime(end_time).time())

    filtered_data = interpolated_data[mask]

    # Proceed with plotting only if there is data to plot
    if not filtered_data.empty:
        # Check for any zero values
        print("Zero values in clear_sky_ghi:", (filtered_data['clear_sky_ghi'] == 0).any())
        # Mask nighttime values for plotting
        filtered_data['masked_ship_radiation'] = filtered_data['ship_radiation'].apply(lambda x: 0 if x < 10 else x)
        filtered_data['masked_interpolated_ssi'] = filtered_data.apply(
            lambda row: 0 if row['ship_radiation'] < 10 else row['interpolated_ssi'], axis=1
        )

        # Replace zero or NaN values in 'clear_sky_ghi' to avoid division by zero
        filtered_data['clear_sky_ghi'].replace(0, np.nan, inplace=True)
        filtered_data.dropna(subset=['clear_sky_ghi'], inplace=True)

        # Calculate CSI for ship radiation and satellite interpolated SSI
        filtered_data['ship_csi'] = filtered_data['masked_ship_radiation'] / filtered_data['clear_sky_ghi']
        filtered_data['sat_csi'] = filtered_data['masked_interpolated_ssi'] / filtered_data['clear_sky_ghi']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot GHI values
        ax1.plot(filtered_data['time'], filtered_data['ship_radiation'], label='Ship GHI', color='blue')
        ax1.plot(filtered_data['time'], filtered_data['interpolated_ssi'], label='Interpolated Satellite GHI', color='red')
        ax1.plot(filtered_data['time'], filtered_data['clear_sky_ghi'], label='Clear Sky GHI', linestyle=':', color='green')
        ax1.set_xlabel('Time [UTC MM-DD HH]')
        ax1.set_ylabel('GHI [W/m^2]')
        ax1.set_title(f'Comparison of Ship GHI, Interpolated Satellite GHI, and Clear Sky GHI for {datetime.strptime(day, "%Y%m%d").strftime("%d/%m/%Y")}')
        ax1.legend()
        ax1.grid(True)

        # Plot CSI values
        ax2.plot(filtered_data['time'], filtered_data['ship_csi'], label='Ship CSI', color='purple')
        ax2.plot(filtered_data['time'], filtered_data['sat_csi'], label='Satellite CSI', color='orange')
        ax2.set_xlabel('Time [UTC MM-DD HH]')
        ax2.set_ylabel('CSI')
        # ax2.set_ylim(0, 1)  # CSI ranges from 0 to 1
        ax2.set_title('Clear Sky Index (CSI) Comparison')
        ax2.legend()
        ax2.grid(True)

        # Calculate and display average CSI values
        avg_ship_csi = filtered_data['ship_csi'].mean()
        avg_sat_csi = filtered_data['sat_csi'].mean()
        print("Average Ship CSI:", avg_ship_csi)
        print("Average Satellite CSI:", avg_sat_csi)
        textstr = f'Average Ship CSI: {avg_ship_csi:.2f}\nAverage Satellite CSI: {avg_sat_csi:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        # Calculate errors
        # errors = calculate_errors(filtered_data)

        # Create a secondary axis for errors
        # ax3 = ax1.twinx()
        # ax3.plot(filtered_data['time'], errors['residuals'], label='Residuals', color='magenta', linestyle='--')
        # ax3.set_ylabel('Residuals [W/m^2]')
        # ax3.legend(loc='upper right')

        # Display error metrics on the plot
        # textstr_errors = f'MBE: {errors["MBE"]:.2f} W/m^2\nRMSE: {errors["RMSE"]:.2f} W/m^2'
        # ax3.text(0.05, 0.1, textstr_errors, transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'figures/ssi_csi_comparison_{day}.png')
        plt.close()
    else:
        print("No data available in the specified time range.")
        
def plot_errors(interpolated_data, day, start_time='15:30', end_time='23:59'):
    # Convert 'time' column to datetime if not already
    interpolated_data['time'] = pd.to_datetime(interpolated_data['time'])

    # Filter data within the specified time range
    if end_time > start_time:
        mask = (interpolated_data['time'].dt.time >= pd.to_datetime(start_time).time()) & \
               (interpolated_data['time'].dt.time <= pd.to_datetime(end_time).time())
    else:  # Over midnight scenario
        mask = (interpolated_data['time'].dt.time >= pd.to_datetime(start_time).time()) | \
               (interpolated_data['time'].dt.time <= pd.to_datetime(end_time).time())

    filtered_data = interpolated_data[mask]

    # Proceed with plotting only if there is data to plot
    if not filtered_data.empty:
        # Calculate errors
        errors = calculate_errors(filtered_data)

        # Create a figure for plotting errors
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot residuals
        ax.plot(filtered_data['time'], errors['residuals'], label='Residuals', color='magenta', linestyle='--')
        ax.set_xlabel('Time [UTC MM-DD HH]')
        ax.set_ylabel('Residuals [W/m^2]')
        ax.set_title(f'Error Analysis for {datetime.strptime(day, "%Y%m%d").strftime("%d/%m/%Y")}')
        ax.legend()
        ax.grid(True)

        # Display MBE and RMSE as horizontal lines
        ax.axhline(y=errors['MBE'], color='blue', linestyle='-', label=f'MBE: {errors["MBE"]:.2f} W/m^2')
        ax.axhline(y=errors['RMSE'], color='red', linestyle='-', label=f'RMSE: {errors["RMSE"]:.2f} W/m^2')
        ax.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'figures/error_analysis_{day}.png')
        plt.close()
    else:
        print("No data available in the specified time range.")

    # Adding labels and title
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Error Metrics')
    ax.set_title('Aggregated Error Metrics by Day')
    ax.legend()

    # Adjust x-ticks to show grouped labels
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'Day {i+1}' for i in range(len(days_formatted))])

    # Save and show plot
    plt.tight_layout()
    plt.savefig('figures/aggregated_errors_by_days.png')
    plt.show()

def add_clear_sky_index(interpolated_data):
    # Calculate CSI for ship radiation
    interpolated_data['ship_csi'] = interpolated_data['ship_radiation'] / interpolated_data['clear_sky_ghi']    
    # Calculate CSI for satellite interpolated SSI
    interpolated_data['sat_csi'] = interpolated_data['interpolated_ssi'] / interpolated_data['clear_sky_ghi']
    
    return interpolated_data

# INFO: Define satellite and processed directories
satellite_dir = 'data/satellite/2017'
processed_dir = 'data/processed'
include_next_day_start = True

sat_day_dirs = [os.path.join(satellite_dir, d) for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]
print("\nSatellite data structure==================================================>")
sat_example = xr.open_dataset('data/satellite/2017/307/20171103000000-OSISAF-RADFLX-01H-GOES13.nc')
print(sat_example)

print(sat_example['lat'].values)
print(sat_example['lon'].values)
# print(sat_example['ssi'].isel(time=0).plot())

ship_data_df = pd.read_csv('data/processed/combined_data_ship.csv')
# Calculate clear sky GHI for the entire DataFrame
ship_data_df = calculate_clear_sky_irradiance(ship_data_df)
# Separate ship data by day
ship_data_by_day = separate_ship_data_by_day(ship_data_df)
print("\nShip data==================================================>")
print(ship_data_df.head())
print()

# INFO: Define desired latitude and longitude range
desired_lat_range = (0, 35)
desired_lon_range = (-130, -105)

# INFO: Save satellite data to csv
# Save_CSV = False
year = '2017'

print("=============================================================================================")
for day_dir in sat_day_dirs:
    # break
    day = os.path.basename(day_dir)
    date = datetime.strptime(f"{year}{day}", "%Y%j").strftime('%Y%m%d')
    # output_file = os.path.join(processed_dir, f'sat_data_{day}.csv')
    print(f"Processing directory: {day}")
    print("=============================================================================================")
    ship_data_day_df = ship_data_by_day[date]
    sat_data_day_df = aggregate_netcdf_to_dataframe_xarray(day_dir, desired_lat_range, desired_lon_range, include_next_day_start)
    # INFO: Save satellite data to csv
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
    
    # Add clear sky index
    interpolated_data = add_clear_sky_index(interpolated_data)
    print("Data with CSI added:")
    print(interpolated_data.head())
    
    # INFO: Save interpolated data to csv
    # output_interpolated_file = os.path.join('data', 'processed', f'interpolated_data_{date}.csv')
    # interpolated_data.to_csv(output_interpolated_file, index=False)
    # print(f"Interpolated data saved to {output_interpolated_file}")
    
    # INFO: Plot comparison or errors or both
    # plot_comparison(interpolated_data, date)
    plot_errors(interpolated_data, date)
    # break
    
    #############################################################################

