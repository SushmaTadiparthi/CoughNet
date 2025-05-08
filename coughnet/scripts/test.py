import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model("D:\Bhanu\CoughNet\coughnet\model\cnn_metadata_image_model.h5")
print("Model loaded successfully.")

# Load test metadata from CSV
test_csv_path = r"D:\Bhanu\CoughNet\coughnet\metadata\test.csv"
print("Loading test metadata...")
df_test = pd.read_csv(test_csv_path)

# Extract the test sample ID
test_id = df_test["id"].values[0]
print(f"Processing Test ID: {test_id}")

# Define the expected features (must match training features exactly)
features = [
    "asthma", "cold", "cough", "diabetes", "diarrhea", "fever", "smoker",
    "loss_of_taste_smell", "wheezing", "body_aches", "chills", "sore_throat",
    "g", "l_c", "a"
]

# Ensure test data has the same structure as training data
metadata_features = df_test[features]

# Normalize metadata (use same scaler as training)
scaler = StandardScaler()
metadata_features_scaled = scaler.transform(metadata_features)

# Reshape for CNN input
metadata_features_reshaped = metadata_features_scaled.reshape(1, metadata_features_scaled.shape[1], 1)

# Function to load spectrogram image
def load_spectrogram_image(image_folder, sample_id, image_size=(128, 128)):
    print("Loading spectrogram image...")
    image_path = os.path.join(image_folder, str(sample_id), "spectrogram")
    if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
        img_file = os.listdir(image_path)[0]  # First spectrogram image
        img = image.load_img(os.path.join(image_path, img_file), target_size=image_size, color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0  # Normalize
        print("Spectrogram loaded successfully.")
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    else:
        print("Spectrogram missing, using placeholder.")
        return np.zeros((1, *image_size, 1))  # Placeholder for missing image

# Load test image
X_test_image = load_spectrogram_image("D:/Bhanu/CoughNet-main/dataset", test_id)

# Make prediction
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
print("=============================")
