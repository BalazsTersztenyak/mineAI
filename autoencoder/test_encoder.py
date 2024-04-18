import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
from models.encoder import Autoencoder

transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Resize images to 640x360 pixels
        transforms.ToTensor(),  # Convert the images to tensors
    ])

# Define your test datapoint
#test_input = torch.randn(1, input_size)  # Adjust input_size according to your model's input shape
test_input = Image.open('test.png').convert('RGB')
test_input = transform(test_input)

# Define your test target (if applicable)
# test_target = ...

# Path to the folder containing the PyTorch checkpoints
checkpoint_folder = "checkpoints"

# Function to load model from a checkpoint
def load_model_from_checkpoint(checkpoint_path):
    model = Autoencoder()  # Initialize your model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Function to test model on a datapoint
def test_model(model, input_data):
    with torch.no_grad():
        output = model(input_data)
        # You can also calculate loss or evaluate accuracy here if you have test targets
        return output

# Iterate over each checkpoint in the folder
for checkpoint_file in os.listdir(checkpoint_folder):
    if checkpoint_file.endswith('.ckpt'):  # Assuming checkpoints have .ckpt extension
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        print(f"Testing checkpoint: {checkpoint_file}")
        
        # Load model from checkpoint
        model = load_model_from_checkpoint(checkpoint_path)
        
        for img in os.listdir('dataset/test_frames'):
            if img.endswith('.png'):
                test_input = Image.open(f'dataset/test_frames/{img}').convert('RGB')
                test_input = transform(test_input)

                # Test model on the datapoint
                output = test_model(model, test_input)
                
                # Print or log the output
                print("Saving output image...")

                # Convert the frame to a PIL Image
                pil_image = transforms.functional.to_pil_image(output)

                # Save the PIL Image as PNG
                pil_image.save(f'dataset/test_frames/{img.split(".")[0]}_{checkpoint_file.split(".")[0]}.png')
