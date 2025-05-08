import os
import pandas as pd
import torchaudio
torchaudio.set_audio_backend("sox_io")

import os
import torchaudio
import pandas as pd

def load_wav_and_labels(train_audio_dir, test_audio_dir, train_label_dir, test_label_dir):
    data = {'train': [], 'test': []}

    # Define a helper function to load files from a given directory
    def load_data(audio_dir, label_dir, split):
        for filename in os.listdir(audio_dir):
            if filename.endswith('.wav'):
                wav_path = os.path.join(audio_dir, filename)
                label_path = os.path.join(label_dir, filename.replace('.wav', '.csv'))

                try:
                    # Load the audio file using torchaudio
                    audio, sr = torchaudio.load(wav_path)  # Much faster than librosa!

                    # Load the label file if it exists
                    label = pd.read_csv(label_path) if os.path.exists(label_path) else None

                    # Append the loaded data to the appropriate split (train/test)
                    data[split].append({
                        'filename': filename,
                        'audio': audio,
                        'sr': sr,
                        'label': label
                    })

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    # Load training data
    load_data(train_audio_dir, train_label_dir, 'train')

    # Load testing data
    load_data(test_audio_dir, test_label_dir, 'test')

    return data

def train_test_split_data(file_paths, labels, test_size=0.2):
    # Since we already have train/test sets, we don't need to split again here
    return file_paths['train'], file_paths['test'], labels['train'], labels['test']
