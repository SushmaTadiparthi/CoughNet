import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define dataset directory (Modify as needed)
dataset_dir = r"<Dataset_Dir>"

# Function to convert an audio file to a spectrogram
def convert_to_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        print(f"ERROR: Failed to process {file_path} - {e}")
        return False

# Get list of folders to process
folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
total_folders = len(folders)

print(f"Total folders found: {total_folders}\n")

# Process each folder
for idx, folder in enumerate(folders, start=1):
    folder_path = os.path.join(dataset_dir, folder)
    spectrogram_dir = os.path.join(folder_path, "spectrogram")
    os.makedirs(spectrogram_dir, exist_ok=True)

    print(f"Processing folder {idx}/{total_folders}: {folder}")

    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    total_files = len(audio_files)

    if total_files == 0:
        print(f"  - No WAV files found in {folder}. Skipping...\n")
        continue

    converted_count = 0

    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        output_path = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}.png")

        if convert_to_spectrogram(file_path, output_path):
            converted_count += 1
            print(f"  - Converted: {file} -> {output_path}")

    print(f"Completed folder {folder}. {converted_count}/{total_files} files converted.\n")
    remaining_folders = total_folders - idx   
    print(f"Remaining folders: {remaining_folders}\n")

print("All folders processed. Spectrogram conversion complete!")
