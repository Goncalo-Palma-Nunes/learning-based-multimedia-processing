import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_mel_spectrogram(tensor_audio, sr=22050, n_mels=128, fmax=8000):
    audio_np = tensor_audio.numpy().squeeze()
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db