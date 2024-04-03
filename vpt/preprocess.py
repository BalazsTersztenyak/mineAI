import pandas as pd
import os

# Directory where your CSV files are located
input_directory = 'data'

# Directory where you want to save modified CSV files
output_directory = 'data'

# List all CSV files in the input directory
csv_files = [file for file in os.listdir(input_directory) if file.endswith('.csv')]

class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initialize the Autoencoder class with encoder and decoder layers using nn.Sequential and nn.Conv2d/nn.ConvTranspose2d.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 45, 80
            nn.ReLU(True)
        )
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
        Perform the forward pass through the encoder and decoder and return the result.
        
        Args:
            x: Input data to be processed.
            
        Returns:
            The output of the forward pass.
        """
        x = self.encoder(x)
        # x = self.decoder(x)
        return x

model = Autoencoder()

# Loop through each CSV file
for file in csv_files:
    # Read CSV file into a DataFrame
    df = pd.read_csv(os.path.join(input_directory, file))
    
    for i, line in iterate(df):
        vec = model.forward(line['pov'])
        df.iloc[i]['pov_vec'] = vec
        if (i > 0):
            prev_line = df.iloc[i-1]
            delta = line - prev_line
            df.iloc[i]['dpos'] = delta['pos']
            df.iloc[i]['dyaw'] = delta['yaw']
            df.iloc[i]['dpitch'] = delta['pitch'] 

        print(df.iloc[i])
        break
    break
    
    # Save the modified DataFrame back to a CSV file
    output_file = os.path.join(output_directory, file)
    df.to_csv(output_file, index=False)
