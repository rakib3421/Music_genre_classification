from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pickle
import os
import pandas as pd
import sys

# Force the stdout to use utf-8 encoding to avoid 'charmap' codec errors
sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your trained model
model = load_model('model/model.keras')  # Update with the path to your model

# Load your scaler and PCA object
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/pca.pkl', 'rb') as f:
    pca = pickle.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3'}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Process the uploaded audio
                audio_features = extract_features(filepath)
                print(f"Extracted features: {audio_features}")  # Debug statement

                # Ensure the input data has the appropriate feature names
                feature_names = list(audio_features.keys())
                a = np.array([list(audio_features.values())])
                data = pd.DataFrame(a, columns=feature_names)

                print(f"Dataframe for scaling: {data}")  # Debug statement

                # Adjust the feature names to match those used in the scaler
                if 'melspectrogram' in feature_names:
                    data = data.rename(columns={'melspectrogram': 'melspectogram'})
                print(f"Renamed DataFrame: {data}")  # Debug statement

                data_scaled = scaler.transform(data)
                print(f"Scaled data: {data_scaled}")  # Debug statement

                data_pca = pca.transform(data_scaled)
                print(f"PCA transformed data: {data_pca}")  # Debug statement

                genre_prediction = model.predict(data_pca)
                print(f"Model prediction: {genre_prediction}")  # Debug statement

                predicted_class = genre_prediction.argmax(axis=1)[0]
                print(f"Predicted class: {predicted_class}")  # Debug statement

                genre_labels = {
                    0: "adhunik",
                    1: "band",
                    2: "hiphop",
                    3: "nazrul",
                    4: "palligeeti",
                    5: "rabindra"
                }
                label = genre_labels[predicted_class]

                return render_template('index.html', audio_path=filename, prediction=label)
            except Exception as e:
                print(f"Error: {e}")  # Debug statement
                return render_template('index.html', error=str(e))
    return render_template('index.html')


def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)

    features = {}
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features['chroma_frequency'] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
    features['rmse'] = np.mean(librosa.feature.rms(y=audio))
    features['melspectogram'] = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr))  # Feature name fixed

    # Updated to use correct tempo extraction method
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = np.mean(tempo)

    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc{i}'] = mfcc

    return features


if __name__ == '__main__':
    app.run(debug=True)