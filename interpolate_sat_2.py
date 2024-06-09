import os
import pandas as pd
import xarray as xr

satellite_dir = 'data/satellite/2017'
sat_day_dirs = [os.path.join(satellite_dir, d) for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]

desired_lat_range = (0, 35)
desired_lon_range = (-105, -130)
# desired_lon_range = (230,255)

def aggregate_netcdf_to_dataframe_xarray(day_dir, desired_lat_range, desired_lon_range):
    nc_files = [os.path.join(day_dir, f) for f in os.listdir(day_dir) if f.endswith('.nc')]
    
    datasets = []
    for nc_file in nc_files:
        ds = xr.open_mfdataset(nc_file, preprocess=lambda ds: ds.assign_coords(time=ds['time']) if 'time' in ds.variables else ds)
        
        # Filter latitude and longitude ranges
        # ds = ds.sel(lat=slice(desired_lat_range[0], desired_lat_range[1]), 
        #             lon=slice(desired_lon_range[0], desired_lon_range[1]))
        
        datasets.append(ds)
    
    # Concatenate datasets along time dimension
    combined_ds = xr.concat(datasets, dim='time')
    
    # Set 'time' as a coordinate
    combined_ds = combined_ds.set_coords('time')
    
    # Convert to DataFrame
    df = combined_ds.to_dataframe()
    
    # Pivot DataFrame to have time as index, lat/lon as columns
    df = df.reset_index().pivot(index='time', columns=['lat', 'lon'], values='ssi')
    
    return df

for day_dir in sat_day_dirs:
    sat_data_day = aggregate_netcdf_to_dataframe_xarray(day_dir, desired_lat_range, desired_lon_range)
    print(sat_data_day.head())
    break