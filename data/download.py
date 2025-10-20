import os
import requests
import zipfile

def download_and_extract(domain, DATA_PATH=None):
    
    BASE_URL = "https://zenodo.org/records/15797226/files"

    zip_path = os.path.join(DATA_PATH, f"{domain}_domain.zip")
    download_url = f"{BASE_URL}/{domain}_domain.zip?download=1"

    # Download the zip file
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract zip contents into DATA_PATH
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)

    # Remove the zip file after extraction
    os.remove(zip_path)

if __name__ == "__main__":
    domain = "NZ"
    DATA_PATH = f'/r/scratch/users/mschillinger/data/cordexbench/{domain}'
    os.makedirs(DATA_PATH, exist_ok=True)

    download_and_extract(domain, DATA_PATH=DATA_PATH)