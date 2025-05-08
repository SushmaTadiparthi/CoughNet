import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import os
import joblib  # For loading pre-trained scaler

# ====== Load the trained model ======
print("Loading model...")
model_path = r"D:\Bhanu\CoughNet\coughnet\model\cnn_metadata_image_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# ====== Load the pre-trained scaler ======
scaler_path = r"D:\Bhanu\CoughNet\coughnet\model\scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
scaler = joblib.load(scaler_path)  # Load trained scaler
print("Scaler loaded successfully.")

# ====== Load test metadata from CSV ======
test_csv_path = r"D:\Bhanu\CoughNet\coughnet\metadata\test.csv"
if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"Test metadata CSV not found: {test_csv_path}")

df_test = pd.read_csv(test_csv_path)
if df_test.empty:
    raise ValueError("Test metadata CSV is empty!")

# Extract the test sample ID
if "id" not in df_test.columns:
    raise KeyError("Column 'id' not found in test metadata CSV!")

test_id = df_test["id"].values[0]
print(f"Processing Test ID: {test_id}")

# ====== Define expected features (Must match training features exactly) ======
features = [
    "asthma", "cold", "cough", "diabetes", "diarrhea", "fever", "smoker",
    "loss_of_taste_smell", "wheezing", "body_aches", "chills", "sore_throat",
    "g", "l_c", "a"
]

# Check if all required features exist in the test CSV
missing_features = [feat for feat in features if feat not in df_test.columns]
if missing_features:
    raise KeyError(f"Missing features in test CSV: {missing_features}")

# Extract metadata features and handle missing values (if any)
metadata_features = df_test[features].fillna(0)  # Replace missing values with 0

# Normalize metadata (use the same scaler from training)
metadata_features_scaled = scaler.transform(metadata_features)

# Reshape for CNN input
metadata_features_reshaped = metadata_features_scaled.reshape(1, metadata_features_scaled.shape[1], 1)

# ====== Function to Load Spectrogram Image ======
def load_spectrogram_image(image_folder, sample_id, image_size=(128, 128)):
    print("Loading spectrogram image...")
    image_path = os.path.join(image_folder, str(sample_id), "spectrogram")

    if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
        img_file = os.listdir(image_path)[0]  # Take the first spectrogram image
        img = image.load_img(os.path.join(image_path, img_file), target_size=image_size, color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0  # Normalize
        print("Spectrogram loaded successfully.")
        return np.expand_dims(img_array, axis=0)  # Add batch dimension

    else:
        print("Spectrogram missing! Using a placeholder image.")
        placeholder_path = r"D:\Bhanu\CoughNet\coughnet\metadata\avg_spectrogram.npy"
        if os.path.exists(placeholder_path):
            avg_spectrogram = np.load(placeholder_path)  # Load precomputed average spectrogram
            return np.expand_dims(avg_spectrogram, axis=0)  # Add batch dimension
        else:
            print("Warning: Placeholder spectrogram not found. Using zero matrix.")
            return np.zeros((1, *image_size, 1))  # Use zero-matrix if placeholder is missing

# Load test spectrogram
X_test_image = load_spectrogram_image("D:/Bhanu/CoughNet-main/dataset", test_id)

# ====== Make Prediction ======
print("Making prediction...")
prediction = model.predict([metadata_features_reshaped, X_test_image])

# Convert prediction to class label
predicted_class = np.argmax(prediction)
confidence_scores = prediction[0]

# Print results
print("\n===== Prediction Result =====")
print(f"Test Sample ID: {test_id}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence Scores: {confidence_scores}")

# Check if prediction confidence is too low
if np.max(confidence_scores) < 0.6:
    print("Warning: Low confidence, prediction might be inaccurate.")

print("=============================")
