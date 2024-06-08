import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def download_files(base_url, start_day, end_day, year):
    total_days = end_day - start_day + 1
    print(f"Starting download for {total_days} days...")
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=10,  # Increase the total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Methods to retry
        backoff_factor=2  # Increase backoff factor to apply between attempts
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for day in tqdm(range(start_day, end_day + 1), desc="Downloading files"):
        url = f"{base_url}/{str(day).zfill(3)}/"
        print(f"Accessing URL: {url}")
        try:
            response = session.get(url, timeout=20)  # Increase timeout to 20 seconds
        except requests.exceptions.RequestException as e:
            print(f"Failed to access {url} with error: {e}")
            continue
        
        if response.status_code != 200:
            print(f"Failed to access {url}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        files_downloaded = 0
        for link in soup.find_all('a'):
            file_name = link.get('href')
            if file_name.endswith('.nc'):
                file_url = url + file_name
                print(f"Attempting to download {file_name} from {file_url}")
                time.sleep(1)  # Delay between downloads
                try:
                    file_response = session.get(file_url, timeout=20)  # Increase timeout to 20 seconds
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {file_name} from {file_url} with error: {e}")
                    continue
                if file_response.status_code == 200:
                    directory_path = f'data/satellite/{year}/{str(day).zfill(3)}'
                    os.makedirs(directory_path, exist_ok=True)
                    with open(f'{directory_path}/{file_name}', 'wb') as f:
                        f.write(file_response.content)
                    files_downloaded += 1
                    print(f"Downloaded {file_name} to {directory_path}/")
                else:
                    print(f"Failed to download {file_name} from {file_url}")
        print(f"Total files downloaded for day {day}: {files_downloaded}")

# Base URL without the day
base_url = "https://osi-saf.ifremer.fr/radflux/l3/west_atlantic_east_pacific/goes/hourly/2017"
# Start and end days
start_day = 288
end_day = 321
# Year
year = 2017

download_files(base_url, start_day, end_day, year)