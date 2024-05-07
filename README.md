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

### DEPENDENCIES: ###

- pandas
- netCDF4 [conda install -c conda-forge netcdf4]
- matplotlib
- cartopy
