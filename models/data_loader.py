import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load all .wav files and their corresponding labels
def load_data_from_folder(data_folder):
    # Create paths for audio and labels
    data_paths = {'train': {'audio': os.path.join(data_folder, 'train_data'),
                            'labels': os.path.join(data_folder, 'train_labels')},
                  'test': {'audio': os.path.join(data_folder, 'test_data'),
                           'labels': os.path.join(data_folder, 'test_labels')}}

    # Initialize lists to hold the file paths and labels
    file_paths = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}

    # Load training data
    for filename in os.listdir(data_paths['train']['audio']):
        if filename.endswith(".wav"):
            file_paths['train'].append(os.path.join(data_paths['train']['audio'], filename))
            label_filename = filename.replace('.wav', '.csv')
            labels['train'].append(os.path.join(data_paths['train']['labels'], label_filename))

    # Load testing data
    for filename in os.listdir(data_paths['test']['audio']):
        if filename.endswith(".wav"):
            file_paths['test'].append(os.path.join(data_paths['test']['audio'], filename))
            label_filename = filename.replace('.wav', '.csv')
            labels['test'].append(os.path.join(data_paths['test']['labels'], label_filename))

    return file_paths, labels

def train_test_split_data(file_paths, labels, test_size=0.2):
    # Since we already have train/test sets, we don't need to split again here
    return file_paths['train'], file_paths['test'], labels['train'], labels['test']
