import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
import librosa
from torch.utils.data import Dataset
import torch

class MelSpectrogramDataset(Dataset):
    def __init__(self, data, n_notes=88, max_len=4000):  # max_len in time frames
        self.data = data
        self.n_notes = n_notes
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel']
        label_df = self.data[idx]['label']
        
        mel_tensor = torch.tensor(mel, dtype=torch.float32)

        # Pad or truncate
        if mel_tensor.shape[1] > self.max_len:
            mel_tensor = mel_tensor[:, :self.max_len]
        else:
            pad_width = self.max_len - mel_tensor.shape[1]
            mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_width))

        mel_tensor = mel_tensor.unsqueeze(0)  # Add channel dim: [1, 128, T]

        # Create multi-hot label vector
        label_vector = torch.zeros(self.n_notes)
        for note in label_df['note'].values:
            if 21 <= note <= 108:
                label_vector[note - 21] = 1.0

        return mel_tensor, label_vector
