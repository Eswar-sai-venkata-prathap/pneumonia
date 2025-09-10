import pandas as pd
import numpy as np
import os

# Define the dataset path
dataset_path = r"D:\pneumonia\chest_xray\train\PNEUMONIA"  # Updated to point to the train folder

# Lists to store data
filenames = []
labels = []

# Collect filenames and labels from pneumonia images directly in the folder
print(f"Scanning directory: {dataset_path}")
files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith('.jpeg')]
print(f"Image files found: {files[:5]}{'...' if len(files) > 5 else ''}")
for file in files:
    filenames.append(os.path.join("train", "PNEUMONIA", file))
    if "virus" in file.lower():
        labels.append("virus")
    elif "bacteria" in file.lower():
        labels.append("bacteria")
    else:
        labels.append("unknown")

print(f"Found {len(filenames)} pneumonia files")
if len(filenames) == 0:
    print("Warning: No pneumonia files found. Please check your dataset path and folder structure.")

# Generate synthetic patient data
age = np.random.randint(0, 100, len(filenames))
fever = []
for label in labels:
    if label == "normal":
        fever.append(np.random.choice([0, 1], p=[0.8, 0.2]))  # 80% no fever
    else:  # virus or bacteria
        fever.append(np.random.choice([0, 1], p=[0.3, 0.7]))  # 70% fever

output_dir = r"D:\pneumonia"
output_csv = os.path.join(output_dir, "synthetic_patient_data.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create DataFrame
data = pd.DataFrame({"filename": filenames, "label": labels, "age": age, "fever": fever})
try:
    data.to_csv(output_csv, index=False)
    print(f"Data generation complete for {len(filenames)} images. Check {output_csv}")
except Exception as e:
    print(f"Error saving CSV: {e}")