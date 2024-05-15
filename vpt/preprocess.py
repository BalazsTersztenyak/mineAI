import pandas as pd
import os
import tarfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import models.encoder as encoder
from tqdm import tqdm
import shutil

# Directory where your CSV files are located
input_directory = 'data/processed_data/val'

# Directory where you want to save modified CSV files
output_directory = 'data/processed_data/val'

checkpoint_number = 299

transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Resize images to 640x360
        transforms.ToTensor(),
    ])

# List all CSV files in the input directory
files = [file for file in os.listdir(input_directory) if file.endswith('.pkl.gz')]
files.sort(key=lambda x: int(x.split('.')[0].split('-')[1]))


print('Loading model...')
model = encoder.Autoencoder()

# Load the trained model
model.load_state_dict(torch.load(f'data/checkpoints/model_{checkpoint_number}.ckpt'))
model.eval()
model = model.cuda()

batch_size = 128

# Loop through each tar.gz file
with tqdm(total=len(files), desc=f"Processing files... ", position=0) as pbar:
    for file in files:
        # Read CSV file into a DataFrame
        df = pd.read_pickle(os.path.join(input_directory, file))
        total_images = len(df.index)
        df.drop('pov_vec', axis=1, inplace=True)
        
        pov_vecs = []

        with tqdm(total=total_images, desc=f"Processing {file}... ", position=1, leave=False) as pbar2:
            for i in range(0, total_images, batch_size):
                batch_end = min(i + batch_size, total_images)
                batch_indices = range(i, batch_end)
                
                # Extract images for the batch
                batch_pov = [df.loc[vec, 'pov'] for vec in batch_indices]
                batch_pov = [Image.fromarray(img, mode='RGB') for img in batch_pov]
                batch_pov = [transform(img).cuda() for img in batch_pov]

                # Stack images into a single tensor
                batch_pov = torch.stack(batch_pov)

                # Forward pass through the model
                with torch.no_grad():
                    batch_vec = model.predict(batch_pov)
                    batch_pov.cpu().detach()

                # Flatten the output vectors
                batch_vec = torch.flatten(batch_vec, start_dim=1).cpu().detach().numpy()
                pov_vecs.extend(batch_vec)

                pitches = [df.loc[idx, 'dpitch'] for idx in batch_indices]
                pitches = torch.tensor(pitches)
                pitches = pitches / 180 + 1
                pitches = pitches.detach().cpu().numpy()
                df.loc[batch_indices, 'dpitch'] = pitches

                yaws = [df.loc[idx, 'dyaw'] for idx in batch_indices]
                yaws = torch.tensor(yaws)
                yaws = yaws / 180 + 1
                yaws = yaws.detach().cpu().numpy()
                df.loc[batch_indices, 'dyaw'] = yaws

                pbar2.update(batch_size)

            df['pov_vec'] = pov_vecs
                
            # Save the modified DataFrame back to a CSV file
            output_file = os.path.join(output_directory, file)
            df.to_pickle(output_file)

            # Remove the original.pkl.gz file
            # os.remove(os.path.join(input_directory, file))
        
        pbar.update(1)

# files = [file for file in os.listdir(output_directory) if file.endswith('.pkl.gz')]
# files.sort(key=lambda x: int(x.split('.')[0].split('-')[1]))

# with tqdm(total=len(files), desc=f"Train-Test split... ", position=0) as pbar:
#     for i in range(len(files)):
#         if i < int(len(files) * 0.8):
#             shutil.move(os.path.join(output_directory, files[i]), os.path.join(output_directory, 'train', files[i]))
#         else:
#             shutil.move(os.path.join(output_directory, files[i]), os.path.join(output_directory, 'val', files[i]))
#         pbar.update(1)
        