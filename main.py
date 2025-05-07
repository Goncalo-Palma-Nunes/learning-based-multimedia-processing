# Import necessary libraries
import os
import torch
from torch.utils.data import DataLoader
from models.data_loader import load_data_from_folder, train_test_split_data
from models.dataset import SpectrogramDataset
from models.cnn import TranscriptionCNN
import torch.optim as optim
import torch.nn as nn
from pydub import AudioSegment
import numpy as np
import librosa  # for transformation (MelSpectrogram)

# Path to the music_net data folder
raw_data_file = '/Users/denis/Library/CloudStorage/GoogleDrive-drfafelgueiras@gmail.com/My Drive/IST/musicnet/'

if os.path.exists(raw_data_file):
    print("Found the folder!")
else:
    print("Path not found.")

# Load data (file_paths and labels)
file_paths, labels = load_data_from_folder(raw_data_file)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split_data(file_paths, labels)

# Define a transformation for the spectrogram using librosa (MelSpectrogram)
def audio_to_melspectrogram(file_path, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    # Load the audio file using pydub
    audio = AudioSegment.from_wav(file_path)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Resample to desired sampling rate (if necessary)
    # You can use librosa to resample
    samples_resampled = librosa.resample(samples, audio.frame_rate, sr)
    
    # Compute MelSpectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=samples_resampled, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to log scale (log-mel spectrogram)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return log_mel_spectrogram

# Prepare train and test datasets and dataloaders
train_dataset = SpectrogramDataset(X_train, y_train, transform=audio_to_melspectrogram)
test_dataset = SpectrogramDataset(X_test, y_test, transform=audio_to_melspectrogram)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model
model = TranscriptionCNN(n_notes=88)  # Adjust for the number of notes (88 for piano keys)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = outputs.round()  # Since this is multi-label, round to 0/1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')