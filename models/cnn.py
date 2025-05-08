import torch
import torch.nn as nn

class TranscriptionCNN(nn.Module):
    def __init__(self, n_notes):
        super(TranscriptionCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 500, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_notes)  # Final layer for multi-label classification
        )

    def forward(self, x):
        x = self.cnn(x)  # Pass through CNN layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected layers
        x = torch.sigmoid(x)  # Apply sigmoid activation to ensure output in [0, 1]
        return x
