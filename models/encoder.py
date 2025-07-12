import torch
import torch.nn as nn
from torchinfo import summary

class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initialize the Autoencoder class with encoder and decoder layers using nn.Sequential and nn.Conv2d/nn.ConvTranspose2d.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 9, 7, stride=5, padding=3),  # (b, 3, 360, 640) -> (b, 9, 72, 128) | 691200 -> 82944
            nn.ReLU(True),
            nn.Conv2d(9, 12, 5, stride=2, padding=2),  # (b, 9, 72, 128) -> (b, 16, 36, 64) | 82944 -> 36864
            nn.ReLU(True),
            nn.Conv2d(12, 15, 3, stride=2, padding=1),  # (b, 16, 36, 64) -> (b, 32, 18, 32) | 36864 -> 18432
            nn.ReLU(True),
            nn.Conv2d(15, 18, 3, stride=2, padding=1),  # (b, 32, 18, 32) -> (b, 64, 9, 16) | 18432 -> 9216
            nn.ReLU(True),
            nn.Flatten(1, -1),
            nn.Linear(18*9*16, 2048),
            # nn.Linear(32*18*32, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 2048),
            # nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(2048, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            nn.Linear(2048, 18*9*16),
            nn.Unflatten(1, (18, 9, 16)),
            nn.ConvTranspose2d(18, 15, 3, stride=2, padding=1, output_padding=1),  # b, 12, 18, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(15, 12, 3, stride=2, padding=1, output_padding=1),  # b, 9, 36, 64
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 9, 5, stride=2, padding=2, output_padding=1),  # b, 6, 72, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(9, 3, 7, stride=5, padding=3, output_padding=4),  # b, 3, 360, 640
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

    def summary(self):
        summary(self, (1, 3, 360, 640))

def load_model_from_checkpoint(checkpoint_path):
    model = Autoencoder()  # Initialize your model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
