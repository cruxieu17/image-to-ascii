import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageToASCIIDataset(Dataset):
    def __init__(self, image_dir, ascii_dir, transform=None):
        self.image_dir = image_dir
        self.ascii_dir = ascii_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]
        self.ascii_files = [f for f in os.listdir(ascii_dir) if f.endswith('txt')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        ascii_path = os.path.join(self.ascii_dir, self.ascii_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        with open(ascii_path, 'r') as f:
            ascii_text = f.read()
        return image, ascii_text