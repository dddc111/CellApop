import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class Labeled_CellDataset(Dataset):
    """Apoptotic  cell image dataset class"""
    
    def __init__(self, json_path, mode='train',transform=None):
        """
        Initialize the dataset
        Args:
            json_path: Json file path
            transform: Data transformation/augmentation
        """
        self.json_path = json_path
        self.transform = transform
        self.mode = mode
        self.samples = load_samples(json_path)
        self.to_tensor = transforms.ToTensor()
        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ])

    def load_samples(self, json_path):
        """
        Load sample data
        Args:
            json_path: Json file path
        Returns:
            samples: List of samples, each containing paths for image and label image
        """
        samples = []
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                img_path = item['img_path']
                label_path = item['label_path']
                samples.append((img_path, label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.mode =='train':
            img = self.transform(img)
        
        return img, self.to_tensor(label)
