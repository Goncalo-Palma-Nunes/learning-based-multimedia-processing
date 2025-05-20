import torch
import torch.nn as nn

class TranscriptionCRNN(nn.Module):
    def __init__(self, n_notes):
        super(TranscriptionCRNN, self).__init__()
        # Convolutional layers
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
        # Recurrent layers
        self.rnn = nn.LSTM(input_size=256 * 16 * 500 // (8 * 8), hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        # Fully connected layers
        # self.fc = nn.Sequential(
        #     nn.Linear(512 * (256 * 16 * 500 // (8 * 8)), 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, n_notes)  # Final layer for multi-label classification
        # )
        self.fc = nn.Linear(512, n_notes + 1)  # n_notes + 1 for CTC blank

    # def forward(self, x):
    #     x = self.cnn(x)
    #     x = x.view(x.size(0), -1)
    #     x = x.unsqueeze(1)
    #     x, _ = self.rnn(x)
    #     x = x[:, -1, :]
    #     x = self.fc(x)
    #     x = torch.sigmoid(x)
    #     return x

    def forward(self, x):
        # x: (batch, channel, height, width)
        x = self.cnn(x)  # (batch, channels, h, w)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.contiguous().view(b, w, c * h)  # (batch, width, features)
        x, _ = self.rnn(x)  # (batch, width, 2*hidden)
        x = self.fc(x)  # (batch, width, n_notes + 1)
        x = x.permute(1, 0, 2)  # (width, batch, n_notes + 1) for CTC loss
        return x

# Example usage
if __name__ == "__main__":
    model = TranscriptionCRNN(n_notes=88)  # Assuming 88 notes for piano
    sample_input = torch.randn(32, 1, 128, 1000)  # Batch size of 32, channels=1, height=128, width=1000
    output = model(sample_input)
    print(output.shape)
