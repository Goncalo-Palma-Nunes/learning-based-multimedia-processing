# Import necessary libraries
import os
import torch
from torch.utils.data                        import DataLoader
from models.data_loader                      import load_wav_and_labels, train_test_split_data
from models.dataset                          import MelSpectrogramDataset
from models.cnn                              import TranscriptionCNN
from models.Checkpoint_print                 import print_header, print_small_header, print_update
from models.spectrogram_mel                  import compute_mel_spectrogram
from models.rnn                              import TranscriptionRNN
import torch.optim                           as optim
import torch.nn                              as nn
from pydub                                   import AudioSegment
import numpy                                 as np
import librosa 
import matplotlib.pyplot                     as plt
import sys

DEFAULT_MUSICNET_PATH = '/Users/denis/Library/CloudStorage/GoogleDrive-drfafelgueiras@gmail.com/My Drive/IST/musicnet' # Default path to MusicNet data

######################################################################################################################
#Step 0: fetching the data for training and testing

def fetch_data(musicnet_path=DEFAULT_MUSICNET_PATH):

    print_header("Finding Folder of Musicnet")

    # Path to the music_net data folder
    raw_data_file = musicnet_path

    if os.path.exists(raw_data_file):
        print(f"Path found: {raw_data_file}")
        if not os.path.isdir(raw_data_file):
            print(f"Path is not a directory: {raw_data_file}")
            print("Please specify the correct path to the musicnet data folder.")
            sys.exit(1)
    else:
        print(f"Path not found: {raw_data_file}")
        print("Please specify the correct path to the musicnet data folder.")
        sys.exit(1)

    return raw_data_file

######################################################################################################################
#Step 1: loading all of the data

def load_split(data, musicnet_path=DEFAULT_MUSICNET_PATH):

    print_header("Loading all the data")

    # Define your specific paths
    train_audio_dir = os.path.join(musicnet_path, 'train_data')
    test_audio_dir = os.path.join(musicnet_path, 'test_data')
    train_label_dir = os.path.join(musicnet_path, 'train_labels')
    test_label_dir = os.path.join(musicnet_path, 'test_labels')
    # train_audio_dir = '/Users/denis/Desktop/IST/S2_24_25/PMBA/musicnet/train_data'
    # test_audio_dir = '/Users/denis/Desktop/IST/S2_24_25/PMBA/musicnet/test_data'
    # train_label_dir = '/Users/denis/Desktop/IST/S2_24_25/PMBA/musicnet/train_labels'
    # test_label_dir = '/Users/denis/Desktop/IST/S2_24_25/PMBA/musicnet/test_labels'

    # Call the function with these paths
    data = load_wav_and_labels(train_audio_dir, test_audio_dir, train_label_dir, test_label_dir)

    # Example to access loaded data for training
    train_data = data['train']
    train_data_audio = []
    train_data_labels = []

    # Loop through each entry in train_data and separate the audio and labels
    for entry in train_data:
        audio = entry['audio']  # This will be the tensor with audio data
        label = entry['label']  # This will be the DataFrame with the labels
        
        # Append the audio and label to their respective lists
        train_data_audio.append(audio)
        train_data_labels.append(label)
    print(f"Number of training samples: {len(train_data)}")
    print(train_data_audio[0])

    # Example to access loaded data for testing
    test_data = data['test']
    test_data_audio = []
    test_data_labels = []

    print(f"Number of testing samples: {len(test_data)}")
    # Loop through each entry in train_data and separate the audio and labels
    for entry in test_data:
        audio = entry['audio']  # This will be the tensor with audio data
        label = entry['label']  # This will be the DataFrame with the labels
        
        # Append the audio and label to their respective lists
        test_data_audio.append(audio)
        test_data_labels.append(label)

    print(test_data_audio[0])

    return train_data_audio, train_data_labels, test_data_audio, test_data_labels


######################################################################################################################

#Step 2: prepare train and test datasets and dataloaders

def prepare_data_and_compute_spectogram(train_data_audio, train_data_labels, test_data_audio, test_data_labels):

    print_header("Applying spectrogram to the data")

    # Lists to store results
    train_mels = []
    test_mels = []

    # Process train data
    for audio_tensor in train_data_audio:
        try:
            mel_spec = compute_mel_spectrogram(audio_tensor)
            train_mels.append(mel_spec)
        except Exception as e:
            print(f"Error processing train sample: {e}")

    # Process test data
    for audio_tensor in test_data_audio:
        try:
            mel_spec = compute_mel_spectrogram(audio_tensor)
            test_mels.append(mel_spec)
        except Exception as e:
            print(f"Error processing test sample: {e}")

    print_update("Plotting Spectrogram of the First 3 samples")

    for i, mel in enumerate(train_mels[:3]):  # Just plot first 3
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel, sr=22050, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Train Sample {i} - Mel Spectrogram')
        plt.tight_layout()
        plt.show()

    # Create a new dataset combining mel spectrograms with labels
    train_dataset = [{'mel': mel, 'label': label} for mel, label in zip(train_mels, train_data_labels)]
    test_dataset  = [{'mel': mel, 'label': label} for mel, label in zip(test_mels, test_data_labels)]

    return train_dataset, test_dataset

######################################################################################################################
#Step 4: CNN model training

def train_CNN(train_dataset, test_dataset):

    print_header("Initialize the model")

    train_dataset = MelSpectrogramDataset(train_dataset, n_notes=88)
    test_dataset = MelSpectrogramDataset(test_dataset, n_notes=88)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


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

    return train_loader, test_loader, model

######################################################################################################################
#Step 5: evalutating model in the test dataset

def evaluate_model(model, test_loader):

    print_header("Evaluate the model")

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


#######################################################################################################################
# Step 6: train BLSTM model
def train_BLSTM(train_dataset, test_dataset):
    print_header("Initialize the model")

    train_dataset = MelSpectrogramDataset(train_dataset, n_notes=88)
    test_dataset = MelSpectrogramDataset(test_dataset, n_notes=88)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = TranscriptionRNN(n_notes=88)  # Adjust for the number of notes (88 for piano keys)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
    
    return train_loader, test_loader, model
######################################################################################################################


if __name__ == "__main__":
    # receive parameter from command line
    if len(sys.argv) > 1:
        if len(sys.argv) > 3:
            print("Too many parameters. Please specify only one of the following as a first parameter:")
            print("1. 'CNN' for a convolutional neural network")
            print("2. 'BLSTM' for a bidirectional long short-term memory network")
            print("3. 'CRNN' for a convolutional recurrent neural network")
            print("The second parameter should be the path to musicnet data folder (can be omitted)")
            sys.exit(1)
        

        param = sys.argv[1]
        if len(sys.argv) == 3:
            musicnet_path = sys.argv[2]
        else:
            musicnet_path = DEFAULT_MUSICNET_PATH
        print(f"Received parameters: {param, musicnet_path}")

        # Fetch data
        musicnet_path = fetch_data(musicnet_path)
        train_data_audio, train_data_labels, test_data_audio, test_data_labels = load_split(musicnet_path)
        train_dataset, test_dataset = prepare_data_and_compute_spectogram(train_data_audio, train_data_labels, test_data_audio, test_data_labels)

        if param == 'CNN':
            train_loader, test_loader, model = train_CNN(train_dataset, test_dataset)
            evaluate_model(model, test_loader)
        elif param == 'BLSTM':
            train_loader, test_loader, model = train_BLSTM(train_dataset, test_dataset)
            evaluate_model(model, test_loader)
        elif param == 'CRNN':
            print("CRNN model training and evaluation not implemented yet.")
        else:
            print("Invalid parameter. Please specify one of the following as a first parameter:")
            print("1. 'CNN' for a convolutional neural network")
            print("2. 'BLSTM' for a bidirectional long short-term memory network")
            print("3. 'CRNN' for a convolutional recurrent neural network")
            print("The second parameter should be the path to musicnet data folder (can be omitted)")
            sys.exit(1)
    else:
        print("No parameters received. Please specify one of the following as a first parameter:")
        print("1. 'CNN' for a convolutional neural network")
        print("2. 'BLSTM' for a bidirectional long short-term memory network")
        print("3. 'CRNN' for a convolutional recurrent neural network")
        print("The second parameter should be the path to musicnet data folder (can be omitted)")
        sys.exit(1)