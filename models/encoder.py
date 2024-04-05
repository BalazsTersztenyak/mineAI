import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initialize the Autoencoder class with encoder and decoder layers using nn.Sequential and nn.Conv2d/nn.ConvTranspose2d.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.Conv2d(12, 16, 3, stride=2, padding=1),  # b, 64, 45, 80
            nn.ReLU(True),
            nn.Flatten(1, -1)
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 45, 80)),
            nn.ConvTranspose2d(16, 12, 3, stride=2, padding=1, output_padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, 3, stride=2, padding=1, output_padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 360, 640
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Perform the forward pass through the encoder and decoder and return the result.
        
        Args:
            x: Input data to be processed.
            
        Returns:
            The output of the forward pass.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @torch.inference_mode
    def predict(self, x):
        """
        Perform the forward pass through the encoder and return the result.
        
        Args:
            x: Input data to be processed.
            
        Returns:
            The output of the forward pass.
        """
        x = self.encoder(x)
        return x