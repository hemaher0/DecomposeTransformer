import papermill as pm
from datetime import datetime
from pathlib import Path
import nbformat
import os
import chardet

os.chdir("../../../")

file_list = [
    "Getting_Started/ipynb/Generate examples/bert-4-128-yahoo/Prune by CI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-4-128-yahoo/Prune by CI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-4-128-yahoo/Prune by CI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-4-128-yahoo/Prune by CI with Head Pruning(60%).ipynb",

    "Getting_Started/ipynb/Generate examples/bert-6-128-yahoo/Prune by CI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-6-128-yahoo/Prune by CI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-6-128-yahoo/Prune by CI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-6-128-yahoo/Prune by CI with Head Pruning(60%).ipynb",
    
    "Getting_Started/ipynb/Generate examples/bert-mini-yahoo/Prune by CI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-mini-yahoo/Prune by CI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-mini-yahoo/Prune by CI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-mini-yahoo/Prune by CI with Head Pruning(60%).ipynb",
    
    "Getting_Started/ipynb/Generate examples/bert-small-yahoo/Prune by CI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-small-yahoo/Prune by CI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-small-yahoo/Prune by CI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-small-yahoo/Prune by CI with Head Pruning(60%).ipynb",
    
    "Getting_Started/ipynb/Generate examples/bert-tiny-yahoo/Prune by CI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-tiny-yahoo/Prune by CI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-tiny-yahoo/Prune by CI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Generate examples/bert-tiny-yahoo/Prune by CI with Head Pruning(60%).ipynb",
]

script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
for file in file_list:
    file_path = Path(file)
    print(f"Processing {file_path}...")
    notebook_start_time = datetime.now()
    try:
        # Try to open with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    
    except UnicodeDecodeError:
        print(f"Encoding issue detected for {file_path}. Attempting to auto-detect and convert to UTF-8.")

        # If UTF-8 fails, detect the encoding and convert to UTF-8
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            original_encoding = result['encoding']
            print(f"Detected encoding: {original_encoding}")

        # Read the file using the detected encoding and convert it to UTF-8
        with open(file_path, 'r', encoding=original_encoding) as f:
            content = f.read()

        # Rewrite the file with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Re-read the notebook after conversion
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    pm.execute_notebook(str(file_path), str(file_path))

    # Record the end time and calculate duration
    notebook_end_time = datetime.now()
    duration = notebook_end_time - notebook_start_time

    # Output the time taken
    print(f"End time: {notebook_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}\n")
