import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageToASCIIDataset
from model import ImageToASCIIModel
from utils import collate_fn

# Hyperparameters
vocab_size = 128  # ASCII characters
embed_size = 256
hidden_size = 512
num_layers = 2
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
image_dir = 'data/images'
ascii_dir = 'data/ascii'
dataset = ImageToASCIIDataset(image_dir, ascii_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageToASCIIModel(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, captions, lengths in dataloader:
            # Prepare inputs
            images = images.to(device)
            captions = captions.to(device)
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False).data

            # Forward pass
            outputs = model(images, captions, lengths)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs)