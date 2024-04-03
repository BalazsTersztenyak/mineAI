import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms

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

class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initialize the Autoencoder class with encoder and decoder layers using nn.Sequential and nn.Conv2d/nn.ConvTranspose2d.

        The encoder takes an input image of shape (3, 360, 640) and outputs an intermediate
        representation of shape (64, 45, 80). The decoder takes this intermediate representation
        as input and outputs a reconstructed image of the same shape as the input.
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 45, 80
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 360, 640
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Perform the forward pass through the encoder and decoder and return the output.

        The input is passed through the encoder network, which extracts features of the input data.
        The features are then passed through the decoder network, which reconstructs the input data from the features.
        The output is the reconstructed input data.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, 3, 360, 640)

        Returns:
            torch.Tensor: Output data of shape (batch_size, 3, 360, 640)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        
        # Test model on the datapoint
        output = test_model(model, test_input)
        
        # Print or log the output
        print("Saving output image...")

        # Convert the frame to a PIL Image
        pil_image = transforms.functional.to_pil_image(output)

        # Save the PIL Image as PNG
        pil_image.save(f'tests/test_{checkpoint_file.split(".")[0]}.png')
        
        # Optionally, you can compare output with your test targets if available
        # loss = calculate_loss(output, test_target)
        # print("Loss:", loss)
