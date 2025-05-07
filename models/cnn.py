import torch
import torch.nn as nn

class TranscriptionCNN(nn.Module):
    def __init__(self, n_notes=88):
        super(TranscriptionCNN, self).__init__()

        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),  # 1 channel input (spectrogram)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pooling across time
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pooling across time again
        )

        # Fully connected layer to predict notes at each time step
        self.fc = None
        self.n_notes = n_notes

    def forward(self, x):
        # Get the dimensions of the input
        batch_size, _, freq_dim, time_dim = x.size()

        # Apply CNN layers
        x = self.cnn(x.unsqueeze(1))  # Add channel dimension: [B, 1, F, T]

        # Dynamically define the fully connected layer
        if self.fc is None:
            # Adjust according to input shape after CNN layers
            fc_input_dim = 128 * (freq_dim // 4) * (time_dim // 4)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_input_dim, self.n_notes),  # Adjust the number of notes
                nn.Sigmoid()  # Outputs probabilities for each note (0 or 1)
            )

        # Pass through the fully connected layer
        x = self.fc(x)
        return x

