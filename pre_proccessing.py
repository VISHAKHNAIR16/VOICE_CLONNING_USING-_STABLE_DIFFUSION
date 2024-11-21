import os
import librosa
import numpy as np
from tqdm import tqdm
import torch


import os
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS"] = "1"

from speechbrain.pretrained import SpeakerRecognition


# Paths
DATASET_PATH = 'VCTK-Corpus-0.92/wav48_silence_trimmed'
OUTPUT_PATH = 'processed_data'
TARGET_SAMPLE_RATE = 22050
N_MELS = 80
MAX_FRAMES = 800  # Maximum number of time frames for padding/truncation

# Load dvector model
dvector_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/speaker_recognition"
)

# Function to process audio
def preprocess_audio(file_path, sample_rate=TARGET_SAMPLE_RATE):
    """
    Processes audio files into mel-spectrograms using librosa.
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Target sample rate for processing.
    Returns:
        np.ndarray: Mel-spectrogram in decibel (dB) scale.
    """
    # Load audio using librosa
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    # Compute mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS, fmax=8000
    )
    
    # Convert to decibel (dB) scale
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

def normalize_spectrogram(mel_spectrogram, min_val=0, max_val=1):
    """
    Normalizes a mel-spectrogram to a specific range [min_val, max_val].
    Args:
        mel_spectrogram (np.ndarray): Input mel-spectrogram.
        min_val (float): Minimum value of the normalized range.
        max_val (float): Maximum value of the normalized range.
    Returns:
        np.ndarray: Normalized mel-spectrogram.
    """
    mel_min = np.min(mel_spectrogram)
    mel_max = np.max(mel_spectrogram)
    normalized_mel = (mel_spectrogram - mel_min) / (mel_max - mel_min)  # Scale to [0, 1]
    normalized_mel = normalized_mel * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
    return normalized_mel

def pad_or_truncate_spectrogram(mel_spectrogram, max_frames=MAX_FRAMES):
    """
    Pads or truncates a mel-spectrogram to a fixed number of time frames.
    Args:
        mel_spectrogram (np.ndarray): Input mel-spectrogram.
        max_frames (int): Desired number of time frames.
    Returns:
        np.ndarray: Padded or truncated mel-spectrogram.
    """
    _, frames = mel_spectrogram.shape
    if frames < max_frames:
        # Pad with zeros
        padding = max_frames - frames
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
    elif frames > max_frames:
        # Truncate
        mel_spectrogram = mel_spectrogram[:, :max_frames]
    return mel_spectrogram

def extract_speaker_embedding(file_path, model=dvector_model):
    """
    Extract speaker embeddings using a pre-trained dvector model.
    Args:
        file_path (str): Path to the audio file.
        model: Pre-trained speaker embedding model.
    Returns:
        np.ndarray: Speaker embedding vector.
    """
    signal, sr = librosa.load(file_path, sr=None)
    signal_tensor = torch.tensor(signal).unsqueeze(0)  # Add batch dimension
    embedding = model.encode_batch(signal_tensor)
    return embedding.squeeze(0).detach().cpu().numpy()  # Convert to NumPy array

# Create output directory for each speaker
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Process dataset
for speaker in tqdm(os.listdir(DATASET_PATH), desc="Processing Speakers"):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if os.path.isdir(speaker_path):  # Ensure it's a directory
        # Create a subdirectory for each speaker in OUTPUT_PATH
        speaker_output_path = os.path.join(OUTPUT_PATH, speaker)
        os.makedirs(speaker_output_path, exist_ok=True)
        
        for file in os.listdir(speaker_path):
            if file.endswith('.flac'):
                file_path = os.path.join(speaker_path, file)
                try:
                    # Preprocess the audio
                    mel = preprocess_audio(file_path)
                    
                    # Normalize the mel-spectrogram
                    mel = normalize_spectrogram(mel, min_val=0, max_val=1)
                    
                    # Pad or truncate the mel-spectrogram
                    mel = pad_or_truncate_spectrogram(mel)
                    
                    # Extract speaker embedding
                    speaker_embedding = extract_speaker_embedding(file_path, model=dvector_model)

                    # Save the mel-spectrogram
                    output_file = os.path.join(speaker_output_path, f"{os.path.splitext(file)[0]}.npy")
                    np.save(output_file, mel)

                    # Save speaker embeddings
                    embedding_file = os.path.join(speaker_output_path, f"{os.path.splitext(file)[0]}_embedding.npy")
                    np.save(embedding_file, speaker_embedding)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

print(f"Processing complete. Processed data saved in '{OUTPUT_PATH}'.")
