import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.edited_images = []
        self.origin_images = []
        for file in os.listdir(os.path.join(self.root_dir, "edited")):
            self.edited_images.append(os.path.join(self.root_dir, "edited", file))
        for file in os.listdir(os.path.join(self.root_dir, "origin")):
            self.origin_images.append(os.path.join(self.root_dir, "origin", file))

    def __len__(self):
        return min(len(self.edited_images), len(self.origin_images))

    def __getitem__(self, idx):
        edited_image = Image.open(self.edited_images[idx])
        origin_image = Image.open(self.origin_images[idx])
        if self.transform:
            edited_image = self.transform(edited_image)
            origin_image = self.transform(origin_image)
        return edited_image, origin_image
