from transformers import Wav2Vec2FeatureExtractor, Data2VecAudioModel
from scipy.signal import resample_poly
import torch
import numpy as np
import librosa


# Load the feature extractor and model
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h")
# model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")

if __name__ == "__main__":
    # Specify the path to save the model locally
    local_model_path = "./data2vec-audio-large-960h"  # Local directory to store the model

    # Download the pretrained model and feature extractor
    model_name = "facebook/data2vec-audio-base-960h"

    # Download model and feature extractor from Hugging Face model hub
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir=local_model_path)
    model = Data2VecAudioModel.from_pretrained(model_name, cache_dir=local_model_path)

    # Now the model and feature extractor are downloaded to the specified local directory
    print(f"Model loaded to: {local_model_path}")

    # Load an audio file and preprocess it
    audio_file = "./001_Neutral_0_x_1_0.wav"  # Replace with your audio file
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)  # Convert to 16kHz mono

    # Extract features using the feature extractor
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state (50 Hz embeddings)
    hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, 768]
    hidden_states = hidden_states.squeeze(0).numpy()  # Remove batch dimension for simplicity

    # Resample to 30 Hz using polyphase resampling
    input_rate = 50  # Model output frame rate
    target_rate = 30  # Desired frame rate

    resampled_embeddings = resample_poly(hidden_states, up=target_rate, down=input_rate, axis=0)

    print(f"Original shape: {hidden_states.shape}")
    print(f"Resampled shape: {resampled_embeddings.shape}")

    # Now `resampled_embeddings` contains the 30 Hz embeddings
