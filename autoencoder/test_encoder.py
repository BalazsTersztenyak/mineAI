import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
from models.encoder import Autoencoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Resize images to 640x360 pixels
        transforms.ToTensor(),  # Convert the images to tensors
    ])

# Define your test datapoint
#test_input = torch.randn(1, input_size)  # Adjust input_size according to your model's input shape
# test_input = Image.open('test.png').convert('RGB')
# test_input = transform(test_input)

# Define your test target (if applicable)
# test_target = ...

# Path to the folder containing the PyTorch checkpoints
checkpoint_folder = "data/checkpoints"

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image
        except:
            print(f"Couldn't load image: {image_path}")

# Function to load model from a checkpoint
def load_model_from_checkpoint(checkpoint_path):
    model = Autoencoder()  # Initialize your model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Function to test model on a datapoint
def test_model(model, input_data):
    # with torch.no_grad():
        criterion = nn.HuberLoss()
        output = model.forward(input_data)
        # You can also calculate loss or evaluate accuracy here if you have test targets
        print(criterion(output, input_data))
        return output

# Assuming you have your dataset loaded into 'images' variable as a list of PIL images
train_dataset = CustomDataset('./data/dataset/test_frames')

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
files = [file for file in os.listdir(checkpoint_folder) if file.endswith('.ckpt')]
files.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
# Iterate over each checkpoint in the folder
for checkpoint_file in files:
    if checkpoint_file.endswith('.ckpt'):  # Assuming checkpoints have .ckpt extension
        n = int(checkpoint_file.split('.')[0].split('_')[1])
        if n > 199 or n%10 != 9:
            continue

        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        print(f"Testing checkpoint: {checkpoint_file}")
        
        # Load model from checkpoint
        model = load_model_from_checkpoint(checkpoint_path).to('cuda')
        
        with tqdm(total=len(train_loader), desc="Testing: ") as pbar:
            for images in train_loader:
                with torch.no_grad():
                    # Test model on the datapoint
                    # outputs = model.forward(images.to('cuda')).to('cpu')
                    outputs = test_model(model, images.to('cuda')).to('cpu')
                
                # Print or log the output
                # print("Saving output image...")

                for i, img in enumerate(outputs):
                    # Convert the frame to a PIL Image
                    # print(torch.mean(images), torch.mean(img), torch.mean(torch.rand(3, 360, 640)))
                    pil_image = transforms.functional.to_pil_image(img)

                    # Save the PIL Image as PNG
                    pil_image.save(f'data/dataset/model_frames/test_{i}_{checkpoint_file.split(".")[0]}.png')

                pbar.update(1)
        model.to('cpu')
        del model
