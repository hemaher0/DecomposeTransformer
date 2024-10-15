import papermill as pm
from datetime import datetime
from pathlib import Path
import nbformat
import chardet
import os

os.chdir("../../../")

variables = {
    "method": [
        "Prune by CI",
    ],
    "model": [
        "bert-4-128-yahoo",
        "bert-6-128-yahoo",
        "bert-mini-yahoo",
        "bert-small-yahoo",
        "bert-tiny-yahoo",
    ],
    "ratio": ["30%", "40%", "50%", "60%"],
}


file_list = []

for method in variables["method"]:
    for model in variables["model"]:
        for ratio in variables["ratio"]:
            file_name = f"experiments/ipynb/{method}/{model}/{method}({ratio}).ipynb"
            file_list.append(file_name)

script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
for file in file_list:
    file_path = Path(file)
    print(f"Processing {file_path}...")
    notebook_start_time = datetime.now()
    try:
        # Try to open with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

    except UnicodeDecodeError:
        print(
            f"Encoding issue detected for {file_path}. Attempting to auto-detect and convert to UTF-8."
        )

        # If UTF-8 fails, detect the encoding and convert to UTF-8
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            original_encoding = result["encoding"]
            print(f"Detected encoding: {original_encoding}")

        # Read the file using the detected encoding and convert it to UTF-8
        with open(file_path, "r", encoding=original_encoding) as f:
            content = f.read()

        # Rewrite the file with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Re-read the notebook after conversion
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    pm.execute_notebook(str(file_path), str(file_path))

    # Record the end time and calculate duration
    notebook_end_time = datetime.now()
    duration = notebook_end_time - notebook_start_time

    # Output the time taken
    print(f"End time: {notebook_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}\n")
