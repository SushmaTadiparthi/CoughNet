import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the processed metadata
print("Loading metadata...")
df = pd.read_csv(r"D:\Bhanu\CoughNet\coughnet\metadata\metadata.csv")
print(f"Metadata loaded. Total rows: {len(df)}")

# Define input features (excluding target labels for testing)
input_features = [
    "asthma", "cold", "cough", "diabetes", "diarrhea", "fever", "smoker", 
    "loss_of_taste_smell", "wheezing", "body_aches", "chills", "sore_throat", "g", "l_c", "a"
]

# Select features and target labels for training
metadata_features = df[input_features]
target = df["overall_status"]

# Normalize the metadata
scaler = StandardScaler()
metadata_features_scaled = scaler.fit_transform(metadata_features)

# Reshape metadata for CNN input
metadata_features_reshaped = metadata_features_scaled.reshape(metadata_features_scaled.shape[0], metadata_features_scaled.shape[1], 1)

# Function to load spectrogram images
def load_spectrogram_images(image_folder, image_size=(128, 128)):
    print("Loading spectrogram images...")
    images = []
    missing_ids = 0
    total_loaded = 0
    for idx in df['id']:
        image_path = os.path.join(image_folder, str(idx), "spectrogram")
        if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
            img_file = os.listdir(image_path)[0]  # Take the first spectrogram image
            img = image.load_img(os.path.join(image_path, img_file), target_size=image_size, color_mode='grayscale')
            img_array = image.img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            total_loaded += 1
        else:
            missing_ids += 1
            images.append(np.zeros((*image_size, 1)))  # Placeholder for missing images
    print(f"Total images loaded: {total_loaded}, Missing images: {missing_ids}")
    return np.array(images)

# Load spectrogram images
image_data = load_spectrogram_images("D:\Bhanu\CoughNet-main\dataset", image_size=(128, 128))

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train_metadata, X_test_metadata, X_train_images, X_test_images, y_train, y_test = train_test_split(
    metadata_features_reshaped, image_data, target, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train_metadata)}, Testing samples: {len(X_test_metadata)}")

# Define the metadata input branch
metadata_input = keras.Input(shape=(X_train_metadata.shape[1], 1), name="metadata_input")
metadata_branch = keras.layers.Conv1D(64, kernel_size=2, activation='relu')(metadata_input)
metadata_branch = keras.layers.MaxPooling1D(pool_size=2)(metadata_branch)
metadata_branch = keras.layers.Conv1D(128, kernel_size=2, activation='relu')(metadata_branch)
metadata_branch = keras.layers.GlobalAveragePooling1D()(metadata_branch)

# Define the image input branch
image_input = keras.Input(shape=(128, 128, 1), name="image_input")
image_branch = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(image_input)
image_branch = keras.layers.MaxPooling2D(pool_size=(2,2))(image_branch)
image_branch = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(image_branch)
image_branch = keras.layers.MaxPooling2D(pool_size=(2,2))(image_branch)
image_branch = keras.layers.Flatten()(image_branch)

# Combine both branches
combined = keras.layers.concatenate([metadata_branch, image_branch])
final_dense = keras.layers.Dense(128, activation='relu')(combined)
final_dense = keras.layers.Dropout(0.5)(final_dense)
final_dense = keras.layers.Dense(64, activation='relu')(final_dense)
output = keras.layers.Dense(4, activation='softmax')(final_dense)

# Create the model
model = keras.Model(inputs=[metadata_input, image_input], outputs=output)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train model
print("Starting training...")
history = model.fit(
    [X_train_metadata, X_train_images], y_train,
    epochs=15, batch_size=8,
    validation_data=([X_test_metadata, X_test_images], y_test)
)
print("Training completed.")

# Save the model
model.save("D:\Bhanu\CoughNet\coughnet\model\cnn_metadata_image_model.h5")
print("Model saved as cnn_metadata_image_model.h5")
