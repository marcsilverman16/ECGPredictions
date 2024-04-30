#!/usr/bin/env python3
"""Download data from PhysioNet."""

import requests
from pathlib import Path
import zipfile

# https://realpython.com/python-download-file-from-url/
url = "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
response = requests.get(url, stream=True)

# Write the URL response to a file.
raw_data_path = Path("..", "raw_data")
# https://stackoverflow.com/a/50110841/8423001
# Make the directory if it doesn't already exist.
# If it already exists, do nothing.
raw_data_path.mkdir(parents=True, exist_ok=True)

raw_ecgs = raw_data_path / "raw_ecgs.zip" 

# Write the file if it does not exist.
if not raw_ecgs.exists():
	with open(raw_ecgs, mode="xb") as file:
		for chunk in response.iter_content(chunk_size=10 * 1024):
			file.write(chunk)

# https://realpython.com/python-zipfile/#getting-started-with-zip-files
with zipfile.ZipFile(raw_ecgs, mode="r") as zippy:
	zippy.extractall(raw_data_path)
	# Delete the zip file now
	Path.unlink(raw_ecgs)

