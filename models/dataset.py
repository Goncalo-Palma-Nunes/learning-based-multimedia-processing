import os
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
import librosa

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, csv_paths, transform=None):
        """
        Args:
            wav_paths (list): List of paths to the .wav files.
            csv_paths (list): List of paths to the CSV files with the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.wav_paths = wav_paths
        self.csv_paths = csv_paths
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieves the spectrogram and corresponding label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (spectrogram, note_labels)
                - spectrogram (Tensor): The spectrogram of the audio.
                - note_labels (Tensor): The piano roll-like matrix of note labels.
        """
        # Load the audio file using pydub
        audio = AudioSegment.from_wav(self.wav_paths[idx])
        
        # Convert audio to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Resample to the desired sample rate (if needed)
        sr = audio.frame_rate
        samples = librosa.resample(samples.astype(float), orig_sr=sr, target_sr=sr)

        # Apply transform to convert waveform to spectrogram if specified
        if self.transform:
            spectrogram = self.transform(samples)
        else:
            # Default: use librosa's mel spectrogram if no transform is specified
            spectrogram = librosa.feature.melspectrogram(y=samples, sr=sr)

        # Load CSV labels and convert to frame-wise labels (piano roll)
        note_labels = self.load_csv_labels(self.csv_paths[idx], spectrogram.shape[1], sr, spectrogram)

        return torch.tensor(spectrogram), note_labels

    def load_csv_labels(self, csv_path, n_frames, sr, spectrogram):
        """
        Loads and processes CSV annotations into a piano roll-like matrix.

        Args:
            csv_path (str): Path to the CSV file containing the note annotations.
            n_frames (int): Number of frames (time steps) in the spectrogram.
            sr (int): Sampling rate of the audio.
            spectrogram (Tensor): The spectrogram of the audio.

        Returns:
            Tensor: A tensor representing the note labels in a piano roll format.
        """
        # Convert CSV note annotations into a piano roll-like matrix
        note_matrix = np.zeros((n_frames, 88))  # 88 keys for piano (A0 to C8)

        # Frame duration (time per frame in seconds)
        frame_duration = spectrogram.shape[1] / sr

        # Process CSV file to extract note start and end times
        with open(csv_path, 'r') as f:
            # Skip the header row if present
            header = f.readline().strip().split(',')
            
            for line in f:
                # Split the CSV line into fields
                fields = line.strip().split(',')
                
                try:
                    start_time = float(fields[0])  # Convert start_time to float
                    end_time = float(fields[1])    # Convert end_time to float
                    instrument = fields[2]         # Instrument (not used for now)
                    note = int(fields[3])          # Convert note to int (MIDI note)
                    start_beat = float(fields[4])  # Start beat (not used for now)
                    end_beat = float(fields[5])    # End beat (not used for now)
                    note_value = float(fields[6])  # Convert note_value to float (velocity/intensity)

                    # Convert start_time and end_time to frame indices
                    start_frame = int(start_time / frame_duration)
                    end_frame = int(end_time / frame_duration)

                    # Convert note to the index for piano roll (MIDI notes start from A0 (21) to C8 (108))
                    note_idx = note - 21  # MIDI notes start from A0 (note 21) to C8 (note 108)
                    if 0 <= note_idx < 88:
                        # You can use the note_value to scale intensity if desired
                        # For now, just mark the note's presence with note_value
                        note_matrix[start_frame:end_frame, note_idx] = note_value  # Note active between start and end frames
                except ValueError:
                    # Skip any rows that can't be parsed properly (optional logging could be added here)
                    continue

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(note_matrix, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.wav_paths)

