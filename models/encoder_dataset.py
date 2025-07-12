from csv import Error
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

torch.manual_seed(42)

TRANSFORM_TEST = transforms.Compose([
    	transforms.Resize(360),
		transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
	])

TRANSFORM_TRAIN = transforms.Compose([
    	transforms.Resize(360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(5),
		transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
	])

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=TRANSFORM_TEST):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            
            return image
        except Error as e:
            print(f"Couldn't load image: {image_path}")
            print(e)
