import os
import shutil
import tarfile
import glob
import subprocess

# Define dataset paths
coswara_data_dir = r'<Dataset_Path>'  # Update with your path
extracted_data_dir = os.path.join(coswara_data_dir, 'dataset')

# Create the extracted dataset directory if it doesn't exist
if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

# Get directories with dataset parts
dirs_extracted = set(map(os.path.basename, glob.glob(f"{extracted_data_dir}\\202*")))
dirs_all = set(map(os.path.basename, glob.glob(f"{coswara_data_dir}\\202*")))
dirs_to_extract = list(dirs_all - dirs_extracted)

for d in dirs_to_extract:
    folder_path = os.path.join(coswara_data_dir, d)
    
    # Step 1: Merge split .tar.gz files
    tar_gz_path = os.path.join(folder_path, f"{d}.tar.gz")
    
    with open(tar_gz_path, "wb") as outfile:
        for part in sorted(glob.glob(os.path.join(folder_path, f"{d}.tar.gz.*"))):
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"✅ Merged {d} split files into a single .tar.gz")

    # Step 2: Extract .tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(os.path.join(extracted_data_dir, d))

    print(f"✅ Extracted {d} successfully!")

print("🎉 Extraction process complete!")