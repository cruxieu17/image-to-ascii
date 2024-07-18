import torch
import torch.nn as nn
from torchvision import models

class ImageToASCIIModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ImageToASCIIModel, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embed_size)  # Change input size to 2048
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions, lengths):
        # Encoder
        features = self.encoder(images)  # Shape: (batch_size, 2048, H, W)
        features = self.avgpool(features)  # Shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Shape: (batch_size, 2048)
        features = self.fc(features)  # Shape: (batch_size, embed_size)

        # Decoder
        embeddings = self.embedding(captions)  # Shape: (batch_size, max_length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # Shape: (batch_size, max_length+1, embed_size)

        # Sort by lengths in descending order
        lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
        embeddings = embeddings[sorted_idx]
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)  # Shape: (batch_size, max_length, hidden_size)
        outputs = self.fc_out(hiddens[0])  # Shape: (batch_size * max_length, vocab_size)
        return outputs