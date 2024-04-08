import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebrityDataset(Dataset):
    def __init__(self, csv_file, img_dir, category_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            category_file (string): Path to the csv file with category names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.categories = pd.read_csv(category_file)
        self.category_to_idx = {row["Category"]: idx for idx, row in self.categories.iterrows()}
        
        # Filter out entries for which the corresponding cropped image does not exist
        self.face_labels = self.face_labels[self.face_labels['File Name'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))]

    def __len__(self):
        return len(self.face_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.face_labels.iloc[idx, 1])
        image = Image.open(img_name)
        category_name = self.face_labels.iloc[idx, 2]
        label = self.category_to_idx[category_name]
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.img_ids = [int(name.split('.')[0]) for name in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_ids[idx]
