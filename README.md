# Satellite Estimated Solar Irradiance Over the Oceans

### Project by Group 7
**Course:** Future Power Supply  
**Degree:** Master of Engineering Physics  
**Institution:** University of Oldenburg  
**Semester:** Summer Semester 2024  

**Tutor:** [Arindam Roy]([#](https://github.com/apodebota))  
**Authors:** [Juan Manuel Boullosa Novo](https://github.com/boujuan), Sofiane Saidi

## Project Structure
This project aims to estimate solar irradiance at the location of a ship using satellite data. The project involves:

1. **Data Acquisition:** Downloading satellite data from the OSI-SAF website and reading ship data from NetCDF files.
2. **Data Processing:** Preprocessing the satellite data, interpolating it to the ship's location and time, and calculating the clear sky index.
3. **Data Analysis:** Comparing the interpolated satellite data with the ship's measurements and analyzing the differences using the clear sky index.
4. **Visualization:** Creating animations and plots to visualize the ship's trajectory, the satellite data, and the comparison between the two.

### Codebase Structure

The codebase is primarily written in Python and consists of several scripts and Jupyter notebooks. Here's an overview of the files and folders:

#### Files

- [`README.md`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/README.md): This file provides an overview of the project, usage instructions, configuration details, and dependencies.
- [`analyse_cdf.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/analyse_cdf.py): This script analysis the general structure of a netCDF file for further exploration.
- [`check_sat_grid_trajectory.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/check_sat_grid_trajectory.py): This script checks and plots the satellite data grid size and compares it to the ship's trajectory hourly to find suitability.
- [`read_data_sat_animation.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/read_data_sat_animation.py): This script reads NetCDF files from a specified directory, extracts relevant variables (latitude, longitude, time, and short-wave radiation), and creates an animation of the satellite data with the ship's trajectory on the map.
- [`read_data_ship_animation.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/read_data_ship_animation.py): This script reads the ship's NetCDF files and extracts the relevant variables (latitude, longitude, time, and short-wave radiation), and creates an animation of the ship's trajectory on the map.
- [`webscrap.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/webscrap.py): This script downloads satellite data from the OSI-SAF website for a specified date range.
- [`interpolate_sat.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/interpolate_sat.py): This script interpolates the satellite data to the ship's location and time, generates a CSV file with the interpolated data and plots the interpolated data on a map.
- [`error_speed_analysis.py`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/blob/main/error_speed_analysis.py): This script analyzes the residuals and the error (MBE, RMSE) between the interpolated satellite data and the ship's measurements for different days and plots the errors.
- Other Python scripts and Jupyter notebooks for specific tasks or exploratory data analysis.

#### Folders

- [`data/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/data): This folder contains the data files used in the project.
  - [`data/satellite/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/data/satellite): This folder stores the downloaded satellite data files in NetCDF format, organized by year and day.
  - [`data/samos/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/data/samos): This folder contains the ship data files in NetCDF format.
  - [`data/processed/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/data/processed): This folder stores processed data files, such as interpolated satellite data and combined ship data in CSV format.
- [`figures/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/figures): This folder contains generated plots and visualizations.
- [`docs/`](https://github.com/boujuan/SatSolarIrradiance_estimationFPS/tree/main/docs): This folder contains some helping documentation of the project.

### Dependencies

The project relies on the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `netCDF4`: For reading and processing NetCDF files.
- `matplotlib`: For creating static visualizations.
- `cartopy`: For creating maps and plotting geographic data.
- `xarray`: For efficient handling of multi-dimensional data arrays.
- `pvlib`: For calculating clear sky irradiance using solar radiation models.
- `requests` and `BeautifulSoup`: For web scraping and downloading satellite data from the OSI-SAF website.

These dependencies can be installed using pip or conda package managers.

### Usage

The main scripts (`read_data.py`, `download_satellite_data.py`, and `process_data.py`) provide instructions on how to use them, including configuration options and required inputs. The `README.md` file also includes usage instructions and configuration details.

[read_data.py]

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
