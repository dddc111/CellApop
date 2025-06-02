import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class UnLabeled_CellDataset(Dataset):
    """Custom cell image dataset class"""
    
    def __init__(self, json_path, transform=None):
        """
        Initialize the dataset
        Args:
            data_dir: Json file path
            transform: Data transformation/augmentation
        """
        self.json_path = json_path
        self.transform = transform
        
        self.samples = load_samples(json_path)
        self.to_tensor = transforms.ToTensor()

        # If no transform is specified, use default data preprocessing
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])

    def load_samples(self, json_path):
        """
        Load sample data
        Args:
            data_dir: Dataset root directory
        Returns:
            samples: List of samples, each containing paths for original image, 
                    augmented image and label image
        """
        samples = []
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                ori_img_path = item['ori_img_path']
                aug_img_path = item['aug_img_path']
                pes_label_path = item['pes_label_path']
                samples.append((ori_img_path, aug_img_path, pes_label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ori_img_path, aug_img_path, pes_label_path = self.samples[idx]
        ori_img = Image.open(ori_img_path).convert('RGB')
        aug_img = Image.open(aug_img_path).convert('RGB')
        pes_label = Image.open(pes_label_path).convert('L')

        # Apply data transformation
        if self.transform:
            ori_img = self.transform(ori_img)
            aug_img = self.transform(aug_img)
        
        return ori_img, aug_img, self.to_tensor(pes_label)
