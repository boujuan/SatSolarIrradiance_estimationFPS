import os
import netCDF4 as NC

def print_nc_structure(nc_obj, indent=0):
    """
    Recursively prints the structure of the netCDF object (group or file).
    """
    # Print variables in the current group
    for var_name, variable in nc_obj.variables.items():
        print(' ' * indent + f"Variable: {var_name}")
        print(' ' * (indent + 2) + f"Dimensions: {variable.dimensions}")
        print(' ' * (indent + 2) + f"Data type: {variable.dtype}")
        print(' ' * (indent + 2) + f"Attributes:")
        for attr_name in variable.ncattrs():
            print(' ' * (indent + 4) + f"{attr_name}: {variable.getncattr(attr_name)}")
        print('---------------------------------')

    # # Recursively print sub-groups
    # for group_name, group in nc_obj.groups.items():
    #     print(' ' * indent + f"Group: {group_name}")
    #     print_nc_structure(group, indent + 2)

def main():
    folder_path = 'samos/netcdf'
    file_list = os.listdir(folder_path)
    file_number = 0
    file_path = os.path.join(folder_path, file_list[file_number])
    print(file_path)

    # Open the NetCDF file
    with NC.Dataset(file_path, 'r') as dataset:
        print(f"File: {file_list[file_number]}")
        print_nc_structure(dataset)
        print(dataset.variables['time'][:])

if __name__ == "__main__":
    main()