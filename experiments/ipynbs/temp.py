import os
import re

percentages = ["30%", "40", "50%", "60%"]
file_list = []
for root, dirs, files in os.walk("."):
    for file in files:
        if any(percentage in file for percentage in percentages):
          if not "checkpoint" in file:
            file_list.append(os.path.join(root, file))

for file_path in file_list:
  match = re.search(r"\((\d+%)\)\.ipynb$", file_path)
  if match:
    percentage = match.group(1)
    
    new_file_name = f"{percentage}.ipynb"
            
    old_file_path = file_path
    new_file_path = os.path.join(os.path.dirname(old_file_path), new_file_name)
    
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} -> {new_file_path}")