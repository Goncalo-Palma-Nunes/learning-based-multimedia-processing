import os
import pandas as pd
import torchaudio
torchaudio.set_audio_backend("sox_io")

import os
import torchaudio
import pandas as pd

def load_wav_and_labels(train_audio_dir, test_audio_dir, train_label_dir, test_label_dir, load_fraction=0.2):
    data = {'train': [], 'test': []}

    def load_data(audio_dir, label_dir, split):
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        wav_files = wav_files[:int(len(wav_files) * load_fraction)]  # Only take the first half

        for filename in wav_files:
            wav_path = os.path.join(audio_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.wav', '.csv'))

            try:
                audio, sr = torchaudio.load(wav_path)
                label = pd.read_csv(label_path) if os.path.exists(label_path) else None
                data[split].append({
                    'filename': filename,
                    'audio': audio,
                    'sr': sr,
                    'label': label
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    load_data(train_audio_dir, train_label_dir, 'train')
    load_data(test_audio_dir, test_label_dir, 'test')

    return data

def train_test_split_data(file_paths, labels, test_size=0.2):
    # Since we already have train/test sets, we don't need to split again here
    return file_paths['train'], file_paths['test'], labels['train'], labels['test']
