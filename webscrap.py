import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

def download_files(base_url, start_day, end_day, year):
    total_days = end_day - start_day + 1
    print(f"Starting download for {total_days} days...")
    
    for day in tqdm(range(start_day, end_day + 1), desc="Downloading files"):
        url = f"{base_url}/{year}/{str(day).zfill(3)}/"
        print(f"Accessing URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access {url}")
            continue
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links on the page
        files_downloaded = 0
        for link in soup.find_all('a'):
            file_name = link.get('href')
            if file_name.endswith('.nc'):  # Check if the link is a .nc file
                file_url = url + file_name
                print(f"Attempting to download {file_name} from {file_url}")
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    # Ensure the directory exists
                    directory_path = f'data/satellite/{year}/{str(day).zfill(3)}'
                    os.makedirs(directory_path, exist_ok=True)
                    # Write the file to the directory
                    with open(f'{directory_path}/{file_name}', 'wb') as f:
                        f.write(file_response.content)
                    files_downloaded += 1
                    print(f"Downloaded {file_name} to {directory_path}/")
                else:
                    print(f"Failed to download {file_name} from {file_url}")
        print(f"Total files downloaded for day {day}: {files_downloaded}")

# Base URL without the day
base_url = "https://osi-saf.ifremer.fr/radflux/l3/east_atlantic_west_indian/meteosat/hourly/2018"
# Start and end days
start_day = 221
end_day = 254
# Year
year = 2018

download_files(base_url, start_day, end_day, year)