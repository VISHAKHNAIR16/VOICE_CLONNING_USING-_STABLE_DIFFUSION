import os
import torch
import numpy as np
import librosa
from torch.utils.data import DataLoader
from tqdm import tqdm
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torch.nn as nn
import soundfile as sf

# ------------------------
# 1. Model Components (Reused from Training Script)
# ------------------------

class MelEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256):
        super(MelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class UNet(nn.Module):
    def __init__(self, input_dim=256):
        super(UNet, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.bottleneck(x)
        return self.up(x)


class MelDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=1):
        super(MelDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Vocoder:
    def __init__(self):
        self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_vocoder")

    def mel_to_audio(self, mel):
        # Ensure mel spectrogram is 3D (batch, n_mels, time_steps)
        if len(mel.shape) == 4:  # e.g., [batch, 1, n_mels, time_steps]
            mel = mel.squeeze(1)
        return self.vocoder.decode_batch(mel)


class StableDiffusionPipeline(nn.Module):
    def __init__(self, latent_dim=256, embedding_dim=512, device="cpu"):
        super(StableDiffusionPipeline, self).__init__()
        self.device = device
        self.encoder = MelEncoder(latent_dim=latent_dim).to(device)
        self.unet = UNet(input_dim=latent_dim).to(device)
        self.decoder = MelDecoder(latent_dim=latent_dim).to(device)
        self.vocoder = Vocoder()
        self.embedding_projector = nn.Linear(embedding_dim, latent_dim).to(device)
        self.tts_model = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_tts_model")

    def forward(self, mel, speaker_embedding, noise_level):
        latent = self.encoder(mel)
        projected_embedding = self.embedding_projector(speaker_embedding).view(latent.size(0), -1, 1, 1)
        latent = latent + projected_embedding.expand_as(latent)
        noise = torch.randn_like(latent)
        noisy_latent = latent + noise * noise_level
        denoised_latent = self.unet(noisy_latent)
        reconstructed_mel = self.decoder(denoised_latent)
        return reconstructed_mel

    def synthesize_audio(self, mel):
        return self.vocoder.mel_to_audio(mel)

    def text_to_mel(self, text, speaker_embedding):
        mel_output = self.tts_model.encode_text(text)[0]
        mel_output = mel_output.unsqueeze(0).to(self.device)
        return mel_output


# ------------------------
# 2. Load Trained Model
# ------------------------

def load_model(model_path, device="cpu"):
    model = StableDiffusionPipeline(device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ------------------------
# 3. Testing Function
# ------------------------

def test_model(model, text, embedding_path, device="cpu"):
    # Load speaker embedding
    speaker_embedding = np.load(embedding_path)
    speaker_embedding_tensor = torch.tensor(speaker_embedding, dtype=torch.float32).to(device)

    # Generate mel spectrogram
    mel_output = model.text_to_mel(text, speaker_embedding_tensor)

    # Synthesize audio
    audio = model.synthesize_audio(mel_output)

    # Ensure audio shape is compatible with soundfile
    audio = audio.squeeze()  # Remove extra dimensions if present
    if len(audio.shape) > 1:
        audio = audio[0]  # Take the first channel if multi-channel

    # Save the generated audio
    output_audio_path = "generated_audio_final134.wav"
    sf.write(output_audio_path, audio.cpu().numpy(), samplerate=22050)
    print(f"Generated audio saved to {output_audio_path}")


# ------------------------
# 4. Run the Test
# ------------------------

if __name__ == "__main__":
    # Define paths
    trained_model_path = "trained_model.pth"
    example_embedding_path = "processed_data/p246/p246_001_mic1_embedding.npy"
    test_text = "hello I am Java."

    # Load model
    device = "cpu"
    model = load_model(trained_model_path, device=device)

    # Run test
    test_model(model, test_text, example_embedding_path, device=device)
