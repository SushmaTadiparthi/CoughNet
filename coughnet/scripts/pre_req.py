import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler

# ====== Load the Training Metadata ======
print("Loading metadata for scaler generation...")
metadata_path = r"D:\Bhanu\CoughNet\coughnet\metadata\metadata.csv"
df = pd.read_csv(metadata_path)

# Define metadata features used in training
features = [
    "asthma", "cold", "cough", "diabetes", "diarrhea", "fever", "smoker",
    "loss_of_taste_smell", "wheezing", "body_aches", "chills", "sore_throat",
    "g", "l_c", "a"
]

# Extract metadata and normalize
metadata_features = df[features].fillna(0)  # Handle missing values
scaler = StandardScaler()
metadata_features_scaled = scaler.fit(metadata_features)  # Fit scaler

# Save the trained scaler
scaler_path = r"D:\Bhanu\CoughNet\coughnet\model\scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at: {scaler_path}")

# ====== Compute and Save the Average Spectrogram ======
def compute_avg_spectrogram(image_folder, image_size=(128, 128)):
    print("Computing average spectrogram...")
    images = []
    for idx in df['id']:
        image_path = os.path.join(image_folder, str(idx), "spectrogram")
        if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
            img_file = os.listdir(image_path)[0]
            img = image.load_img(os.path.join(image_path, img_file), target_size=image_size, color_mode='grayscale')
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)

    if len(images) == 0:
        print("No spectrograms found! Using a black placeholder.")
        avg_spectrogram = np.zeros((*image_size, 1))
    else:
        avg_spectrogram = np.mean(images, axis=0)

    return avg_spectrogram

# Compute and save the average spectrogram
spectrogram_folder = r"D:\Bhanu\CoughNet-main\dataset"
avg_spectrogram = compute_avg_spectrogram(spectrogram_folder)

# Save the average spectrogram
avg_spectrogram_path = r"D:\Bhanu\CoughNet\coughnet\metadata\avg_spectrogram.npy"
np.save(avg_spectrogram_path, avg_spectrogram)
print(f"Average spectrogram saved at: {avg_spectrogram_path}")
