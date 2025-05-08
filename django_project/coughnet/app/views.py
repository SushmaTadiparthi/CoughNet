from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib  # For loading pre-trained scaler

# === Function to Convert Audio to Spectrogram ===
def convert_to_spectrogram(file_path, output_path):
    """Convert an audio file to a mel spectrogram image."""
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
        print(f"❌ ERROR: Failed to process {file_path} - {e}")
        return False

# === Main Page View ===
def main_page(request):
    metadata_fields = [
        "asthma", "cold", "cough", "diabetes", "diarrhea", "fever",
        "smoker", "loss_of_taste_smell", "wheezing", "body_aches", "chills", "sore_throat"
    ]

    metadata_submitted = False
    audio_uploaded = False
    prediction_result = None

    print("🔵 Request method:", request.method)

    if request.method == 'POST':
        # === Retrieve Metadata from Form ===
        metadata = {field: int(request.POST.get(field, 0)) for field in metadata_fields}
        metadata["g"] = float(request.POST.get("g", 0))  
        metadata["l_c"] = float(request.POST.get("l_c", 0.0))  
        metadata["a"] = int(request.POST.get("a", 0))

        print("🟢 Metadata received:", metadata)
        metadata_submitted = True
        
        # === Handle Audio Upload ===
        fs = FileSystemStorage(location='media/audio/')
        audio_files = request.FILES.getlist('audio_files')
        audio_paths = []

        print("🟡 Number of audio files uploaded:", len(audio_files))

        if 1 <= len(audio_files) <= 3:
            for audio in audio_files:
                if audio.size <= 10 * 1024 * 1024:  # Ensure file size is below 10MB
                    filename = fs.save(audio.name, audio)
                    audio_path = os.path.join(fs.location, filename)
                    spectrogram_path = os.path.join('media/spectrograms', f"{filename}.png")

                    if convert_to_spectrogram(audio_path, spectrogram_path):
                        print(f"✅ Spectrogram saved at: {spectrogram_path}")
                        audio_paths.append(spectrogram_path)
                    else:
                        print(f"❌ Failed to create spectrogram for: {audio_path}")

            audio_uploaded = True

        print("🟣 Audio uploaded:", audio_uploaded)

        # === Run Prediction if Both Metadata and Audio Are Provided ===
        if metadata_submitted and audio_uploaded:
            prediction_result = run_prediction(metadata, audio_paths)
            print("🔴 Prediction Result:", prediction_result)

    return render(request, 'app/main_page.html', {
        'metadata_fields': metadata_fields,
        'metadata_submitted': metadata_submitted,
        'audio_uploaded': audio_uploaded,
        'prediction_result': prediction_result
    })

# === Prediction Function ===
def run_prediction(metadata, spectrogram_paths):
    try:
        BASE_DIR = settings.BASE_DIR  # Get Django's base directory dynamically

        # === Load the Trained Model ===
        print("🟢 Loading model...")
        model_path = r"C:\Users\vamsi\OneDrive\Desktop\CoughNet\coughnet\model\cnn_metadata_image_model.h5" 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        # === Load the Pre-trained Scaler ===
        print("🟢 Loading pre-trained scaler...")
        scaler_path = (r"C:\Users\vamsi\OneDrive\Desktop\CoughNet\coughnet\model\scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully!")

        # === Prepare Metadata Features ===
        metadata_df = pd.DataFrame([metadata])
        print("🔵 Raw metadata:\n", metadata_df)
        metadata_scaled = scaler.transform(metadata_df)
        print("🟢 Scaled metadata:\n", metadata_scaled)
        metadata_reshaped = metadata_scaled.reshape(1, metadata_scaled.shape[1], 1)
        print("🟣 Metadata shape:", metadata_reshaped.shape)  # Expected shape: (1, num_features, 1)

        # === Load Spectrogram Image ===
        if spectrogram_paths:
            spectrogram_path = spectrogram_paths[0]
            if not os.path.exists(spectrogram_path):
                raise FileNotFoundError(f"Spectrogram image not found: {spectrogram_path}")

            print("🟢 Loading spectrogram image:", spectrogram_path)
            img = image.load_img(spectrogram_path, target_size=(128, 128), color_mode='grayscale')
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        else:
            print("❌ Spectrogram missing! Using a placeholder image.")
            placeholder_path = (r"C:\Users\vamsi\OneDrive\Desktop\CoughNet\coughnet\modelavg_spectrogram.npy")
            if os.path.exists(placeholder_path):
                avg_spectrogram = np.load(placeholder_path)
                img_array = np.expand_dims(avg_spectrogram, axis=0)
            else:
                print("⚠️ Warning: Placeholder spectrogram not found. Using zero matrix.")
                img_array = np.zeros((1, 128, 128, 1))

        print("🟣 Spectrogram shape:", img_array.shape)  # Expected shape: (1, 128, 128, 1)

        # === Run Prediction ===
        print("🟣 Running prediction...")
        prediction = model.predict([metadata_reshaped, img_array])
        print("✅ Prediction completed!")

        # === Process Prediction Output ===
        confidence_scores = prediction[0] * 100  # Convert to percentage
        predicted_class = np.argmax(confidence_scores)

        class_labels = ["Healthy", "HMPV", "H3N2", "COVID-19"]
        predicted_label = class_labels[predicted_class]

        confidence_text = ", ".join(
            [f"{label}: {score:.2f}%" for label, score in zip(class_labels, confidence_scores)]
        )

        return f"Predicted Condition: {predicted_label} | Confidence Scores: {confidence_text}"

    except Exception as e:
        print("❌ Prediction failed:", e)
        return f"Prediction failed: {e}"
